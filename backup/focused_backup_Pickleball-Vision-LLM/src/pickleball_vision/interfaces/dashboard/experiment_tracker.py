import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ExperimentTracker:
    def __init__(self, tracking_uri: str = "file:./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments."""
        experiments = self.client.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            }
            for exp in experiments
        ]
    
    def get_runs(self, experiment_id: str) -> pd.DataFrame:
        """Get all runs for an experiment."""
        runs = self.client.search_runs(experiment_id)
        data = []
        for run in runs:
            data.append({
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                **run.data.metrics,
                **run.data.params
            })
        return pd.DataFrame(data)
    
    def get_model_versions(self, model_name: str) -> pd.DataFrame:
        """Get all versions of a model."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        data = []
        for version in versions:
            data.append({
                "version": version.version,
                "status": version.status,
                "stage": version.current_stage,
                "run_id": version.run_id
            })
        return pd.DataFrame(data)
    
    def plot_metrics(self, runs_df: pd.DataFrame, metric: str) -> px.Figure:
        """Plot metric over time."""
        fig = px.line(
            runs_df,
            x="start_time",
            y=metric,
            title=f"{metric} over time"
        )
        return fig
    
    def plot_parameter_importance(self, runs_df: pd.DataFrame, metric: str) -> px.Figure:
        """Plot parameter importance for a metric."""
        # Calculate correlation between parameters and metric
        params = [col for col in runs_df.columns if col not in ["run_id", "start_time", "end_time", "status"]]
        correlations = runs_df[params + [metric]].corr()[metric].drop(metric)
        
        # Create bar plot
        fig = px.bar(
            x=correlations.index,
            y=correlations.values,
            title=f"Parameter importance for {metric}"
        )
        return fig
    
    def plot_metrics_comparison(self, runs_df: pd.DataFrame, metrics: List[str]) -> px.Figure:
        """Plot comparison of multiple metrics."""
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=runs_df["start_time"],
                y=runs_df[metric],
                name=metric,
                mode="lines+markers"
            ))
        fig.update_layout(
            title="Metrics Comparison",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified"
        )
        return fig
    
    def plot_confusion_matrix(self, runs_df: pd.DataFrame) -> px.Figure:
        """Plot confusion matrix heatmap."""
        # Extract confusion matrix from metrics
        cm = np.array([
            [runs_df["true_negatives"].mean(), runs_df["false_positives"].mean()],
            [runs_df["false_negatives"].mean(), runs_df["true_positives"].mean()]
        ])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Negative", "Positive"],
            y=["Negative", "Positive"],
            title="Confusion Matrix"
        )
        return fig
    
    def plot_roc_curve(self, runs_df: pd.DataFrame) -> px.Figure:
        """Plot ROC curve."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=runs_df["fpr"],
            y=runs_df["tpr"],
            name="ROC",
            mode="lines"
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random",
            mode="lines",
            line=dict(dash="dash")
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        return fig
    
    def plot_learning_curves(self, runs_df: pd.DataFrame) -> px.Figure:
        """Plot learning curves."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=runs_df["epoch"],
            y=runs_df["train_loss"],
            name="Training Loss",
            mode="lines"
        ))
        fig.add_trace(go.Scatter(
            x=runs_df["epoch"],
            y=runs_df["val_loss"],
            name="Validation Loss",
            mode="lines"
        ))
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )
        return fig
    
    def plot_parameter_correlation(self, runs_df: pd.DataFrame) -> px.Figure:
        """Plot parameter correlation heatmap."""
        params = [col for col in runs_df.columns if col not in ["run_id", "start_time", "end_time", "status"]]
        corr_matrix = runs_df[params].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Parameter 1", y="Parameter 2", color="Correlation"),
            title="Parameter Correlation Matrix"
        )
        return fig

def main():
    st.title("Pickleball Vision Experiment Tracker")
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Experiments", "Models", "Analysis", "Visualizations"])
    
    if page == "Experiments":
        st.header("Experiments")
        
        # Get experiments
        experiments = tracker.get_experiments()
        exp_df = pd.DataFrame(experiments)
        st.dataframe(exp_df)
        
        # Select experiment
        selected_exp = st.selectbox(
            "Select experiment",
            exp_df["name"].tolist()
        )
        
        if selected_exp:
            # Get runs
            exp_id = exp_df[exp_df["name"] == selected_exp]["experiment_id"].iloc[0]
            runs_df = tracker.get_runs(exp_id)
            
            # Display runs
            st.subheader("Runs")
            st.dataframe(runs_df)
            
            # Plot metrics
            st.subheader("Metrics")
            metric = st.selectbox("Select metric", runs_df.columns)
            if metric:
                fig = tracker.plot_metrics(runs_df, metric)
                st.plotly_chart(fig)
    
    elif page == "Models":
        st.header("Model Registry")
        
        # Get models
        models = [model.name for model in tracker.client.search_registered_models()]
        selected_model = st.selectbox("Select model", models)
        
        if selected_model:
            # Get versions
            versions_df = tracker.get_model_versions(selected_model)
            st.dataframe(versions_df)
    
    elif page == "Analysis":
        st.header("Analysis")
        
        # Get experiments
        experiments = tracker.get_experiments()
        exp_df = pd.DataFrame(experiments)
        selected_exp = st.selectbox(
            "Select experiment for analysis",
            exp_df["name"].tolist()
        )
        
        if selected_exp:
            # Get runs
            exp_id = exp_df[exp_df["name"] == selected_exp]["experiment_id"].iloc[0]
            runs_df = tracker.get_runs(exp_id)
            
            # Parameter importance
            st.subheader("Parameter Importance")
            metric = st.selectbox("Select metric for analysis", runs_df.columns)
            if metric:
                fig = tracker.plot_parameter_importance(runs_df, metric)
                st.plotly_chart(fig)
    
    else:  # Visualizations
        st.header("Advanced Visualizations")
        
        # Get experiments
        experiments = tracker.get_experiments()
        exp_df = pd.DataFrame(experiments)
        selected_exp = st.selectbox(
            "Select experiment for visualization",
            exp_df["name"].tolist()
        )
        
        if selected_exp:
            # Get runs
            exp_id = exp_df[exp_df["name"] == selected_exp]["experiment_id"].iloc[0]
            runs_df = tracker.get_runs(exp_id)
            
            # Visualization type
            viz_type = st.selectbox(
                "Select visualization type",
                ["Metrics Comparison", "Confusion Matrix", "ROC Curve", 
                 "Learning Curves", "Parameter Correlation"]
            )
            
            if viz_type == "Metrics Comparison":
                metrics = st.multiselect(
                    "Select metrics to compare",
                    runs_df.columns
                )
                if metrics:
                    fig = tracker.plot_metrics_comparison(runs_df, metrics)
                    st.plotly_chart(fig)
            
            elif viz_type == "Confusion Matrix":
                fig = tracker.plot_confusion_matrix(runs_df)
                st.plotly_chart(fig)
            
            elif viz_type == "ROC Curve":
                fig = tracker.plot_roc_curve(runs_df)
                st.plotly_chart(fig)
            
            elif viz_type == "Learning Curves":
                fig = tracker.plot_learning_curves(runs_df)
                st.plotly_chart(fig)
            
            else:  # Parameter Correlation
                fig = tracker.plot_parameter_correlation(runs_df)
                st.plotly_chart(fig)

if __name__ == "__main__":
    main() 