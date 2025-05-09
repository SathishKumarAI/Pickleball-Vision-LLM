import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class AdvancedVisualizations:
    def __init__(self):
        self.court_dimensions = {
            'length': 44,  # feet
            'width': 20,   # feet
            'net_height': 3  # feet
        }
    
    def create_3d_court_visualization(self, 
                                    shot_data: pd.DataFrame,
                                    player_positions: Dict[str, List[Tuple[float, float, float]]]) -> go.Figure:
        """Create 3D visualization of court with shots and player movements."""
        fig = go.Figure()
        
        # Add court surface
        court_x = [0, 0, self.court_dimensions['length'], self.court_dimensions['length'], 0]
        court_y = [0, self.court_dimensions['width'], self.court_dimensions['width'], 0, 0]
        court_z = [0, 0, 0, 0, 0]
        
        fig.add_trace(go.Scatter3d(
            x=court_x,
            y=court_y,
            z=court_z,
            mode='lines',
            name='Court',
            line=dict(color='green', width=2)
        ))
        
        # Add net
        net_x = [self.court_dimensions['length']/2] * 2
        net_y = [0, self.court_dimensions['width']]
        net_z = [0, self.court_dimensions['net_height']]
        
        fig.add_trace(go.Scatter3d(
            x=net_x,
            y=net_y,
            z=net_z,
            mode='lines',
            name='Net',
            line=dict(color='red', width=2)
        ))
        
        # Add shot trajectories
        for _, shot in shot_data.iterrows():
            fig.add_trace(go.Scatter3d(
                x=shot['trajectory_x'],
                y=shot['trajectory_y'],
                z=shot['trajectory_z'],
                mode='lines+markers',
                name=f'Shot {shot["shot_id"]}',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
        
        # Add player movements
        for player, positions in player_positions.items():
            positions = np.array(positions)
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines+markers',
                name=f'Player {player}',
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title='3D Court Visualization',
            scene=dict(
                xaxis_title='Length (ft)',
                yaxis_title='Width (ft)',
                zaxis_title='Height (ft)',
                aspectmode='data'
            )
        )
        
        return fig
    
    def create_shot_heatmap(self, shot_data: pd.DataFrame) -> go.Figure:
        """Create heatmap of shot placements."""
        # Create 2D histogram of shot placements
        x_bins = np.linspace(0, self.court_dimensions['length'], 20)
        y_bins = np.linspace(0, self.court_dimensions['width'], 10)
        
        H, xedges, yedges = np.histogram2d(
            shot_data['placement_x'],
            shot_data['placement_y'],
            bins=[x_bins, y_bins]
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=H.T,
            x=xedges[:-1],
            y=yedges[:-1],
            colorscale='Viridis',
            showscale=True
        ))
        
        # Add court boundaries
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=self.court_dimensions['length'],
            y1=self.court_dimensions['width'],
            line=dict(color="white"),
            fillcolor="rgba(0,0,0,0)"
        )
        
        fig.update_layout(
            title='Shot Placement Heatmap',
            xaxis_title='Court Length (ft)',
            yaxis_title='Court Width (ft)'
        )
        
        return fig
    
    def create_player_movement_analysis(self, 
                                      player_data: pd.DataFrame,
                                      time_window: int = 5) -> go.Figure:
        """Create visualization of player movement patterns."""
        fig = go.Figure()
        
        # Calculate movement metrics
        player_data['speed'] = np.sqrt(
            np.diff(player_data['x'])**2 + 
            np.diff(player_data['y'])**2
        )
        
        player_data['acceleration'] = np.diff(player_data['speed'])
        
        # Create subplots
        fig = go.Figure()
        
        # Add speed trace
        fig.add_trace(go.Scatter(
            x=player_data['timestamp'],
            y=player_data['speed'],
            mode='lines',
            name='Speed',
            line=dict(color='blue')
        ))
        
        # Add acceleration trace
        fig.add_trace(go.Scatter(
            x=player_data['timestamp'],
            y=player_data['acceleration'],
            mode='lines',
            name='Acceleration',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Player Movement Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Speed (ft/s)',
            yaxis2=dict(
                title='Acceleration (ft/s²)',
                overlaying='y',
                side='right'
            )
        )
        
        return fig
    
    def create_shot_effectiveness_chart(self, shot_data: pd.DataFrame) -> go.Figure:
        """Create visualization of shot effectiveness by type."""
        # Calculate average effectiveness by shot type
        effectiveness = shot_data.groupby('shot_type')['effectiveness_score'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure()
        
        # Add effectiveness bars
        fig.add_trace(go.Bar(
            x=effectiveness['shot_type'],
            y=effectiveness['mean'],
            error_y=dict(
                type='data',
                array=effectiveness['std'],
                visible=True
            ),
            name='Effectiveness'
        ))
        
        fig.update_layout(
            title='Shot Effectiveness by Type',
            xaxis_title='Shot Type',
            yaxis_title='Effectiveness Score',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_rally_analysis(self, rally_data: pd.DataFrame) -> go.Figure:
        """Create visualization of rally patterns and statistics."""
        fig = go.Figure()
        
        # Add rally duration distribution
        fig.add_trace(go.Histogram(
            x=rally_data['duration'],
            name='Rally Duration',
            nbinsx=20
        ))
        
        fig.update_layout(
            title='Rally Duration Distribution',
            xaxis_title='Duration (s)',
            yaxis_title='Count'
        )
        
        return fig 