import streamlit as st
import plotly.graph_objects as go
import json
import requests
import pandas as pd
from datetime import datetime
import os
from typing import Dict, Any, List

# Configure Streamlit page
st.set_page_config(
    page_title="Pickleball Vision Analytics",
    page_icon="🎾",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

def login():
    """Login form for user authentication."""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            try:
                response = requests.post(
                    f"{API_URL}/token",
                    data={"username": username, "password": password}
                )
                if response.status_code == 200:
                    token = response.json()["access_token"]
                    st.session_state["token"] = token
                    st.session_state["username"] = username
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
            except Exception as e:
                st.error(f"Error connecting to server: {str(e)}")

def register():
    """Registration form for new users."""
    st.title("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
                return
                
            try:
                response = requests.post(
                    f"{API_URL}/users/",
                    json={
                        "username": username,
                        "email": email,
                        "password": password
                    }
                )
                if response.status_code == 200:
                    st.success("Registration successful! Please login.")
                    st.session_state["page"] = "login"
                    st.experimental_rerun()
                else:
                    st.error("Registration failed")
            except Exception as e:
                st.error(f"Error connecting to server: {str(e)}")

def load_data() -> Dict[str, Any]:
    """Load sample data for demonstration."""
    return {
        "shots": [
            {
                "player_id": "1",
                "shot_type": "serve",
                "placement_x": 0.5,
                "placement_y": 0.8,
                "speed": 45,
                "spin": 0.2,
                "timestamp": datetime.now().isoformat(),
                "effectiveness_score": 0.8
            }
        ],
        "players": [
            {
                "id": "1",
                "name": "Player 1",
                "skill_level": "intermediate"
            }
        ],
        "rallies": [
            {
                "duration": 15,
                "winner_team": "team1",
                "ending_type": "winner",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat()
            }
        ],
        "player_positions": [
            {
                "player_id": "1",
                "x": 0.5,
                "y": 0.5,
                "timestamp": datetime.now().isoformat(),
                "speed": 5
            }
        ]
    }

def analyze_strategy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze strategy using the API."""
    try:
        response = requests.post(
            f"{API_URL}/api/analyze",
            json=data,
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error analyzing strategy")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return {}

def get_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get visualizations from the API."""
    try:
        response = requests.post(
            f"{API_URL}/api/visualize",
            json=data,
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error getting visualizations")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return {}

def get_user_games() -> List[Dict[str, Any]]:
    """Get user's games from the API."""
    try:
        response = requests.get(
            f"{API_URL}/api/games/",
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error getting games")
            return []
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return []

def get_game_analysis(game_id: str) -> Dict[str, Any]:
    """Get analysis for specific game."""
    try:
        response = requests.get(
            f"{API_URL}/api/games/{game_id}/analysis",
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error getting game analysis")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return {}

def get_advanced_visualizations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get advanced visualizations from the API."""
    try:
        response = requests.post(
            f"{API_URL}/api/advanced-visualizations",
            json=data,
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error getting advanced visualizations")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return {}

def get_game_advanced_analysis(game_id: str) -> Dict[str, Any]:
    """Get advanced analysis for specific game."""
    try:
        response = requests.get(
            f"{API_URL}/api/games/{game_id}/advanced-analysis",
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error getting game advanced analysis")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return {}

def get_player_advanced_stats(player_id: str) -> Dict[str, Any]:
    """Get advanced statistics for specific player."""
    try:
        response = requests.get(
            f"{API_URL}/api/players/{player_id}/advanced-stats",
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Error getting player advanced stats")
            return {}
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")
        return {}

def show_strategy_analysis(analysis: Dict[str, Any], visualizations: Dict[str, Any]):
    """Display strategy analysis visualizations."""
    st.header("Strategy Analysis")
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Metrics")
        metrics = analysis.get("metrics", {})
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.2f}")
    
    # Display visualizations
    with col2:
        st.subheader("Visualizations")
        for viz_name, viz_json in visualizations.items():
            fig = go.Figure(json.loads(viz_json))
            st.plotly_chart(fig, use_container_width=True)

def show_team_analysis(team_data: Dict[str, Any]):
    """Display team analysis visualizations."""
    st.header("Team Analysis")
    
    # Get team visualizations
    try:
        response = requests.post(
            f"{API_URL}/api/team-analysis",
            json={"team_data": team_data},
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        if response.status_code == 200:
            team_viz = response.json()
            
            # Display team visualizations
            for viz_name, viz_json in team_viz.items():
                fig = go.Figure(json.loads(viz_json))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Error getting team analysis")
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")

def show_performance_metrics(games: List[Dict[str, Any]]):
    """Display performance metrics for user's games."""
    st.header("Performance Metrics")
    
    # Convert games to DataFrame
    df = pd.DataFrame(games)
    
    # Display game statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games", len(games))
    with col2:
        avg_duration = df["duration"].mean()
        st.metric("Average Duration", f"{avg_duration:.1f} minutes")
    with col3:
        locations = df["location"].nunique()
        st.metric("Unique Locations", locations)
    
    # Display game history
    st.subheader("Game History")
    st.dataframe(df)

def show_advanced_analysis(data: Dict[str, Any], visualizations: Dict[str, Any]):
    """Display advanced analysis visualizations."""
    st.header("Advanced Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Shot Analysis",
        "Serve Analysis",
        "Dink Analysis",
        "Movement Analysis",
        "Rally Analysis",
        "Performance Trends",
        "Combined View"
    ])
    
    with tab1:
        st.subheader("Shot Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Shot heatmap
            if "shot_heatmap" in visualizations:
                fig = go.Figure(json.loads(visualizations["shot_heatmap"]))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Shot patterns
            if "shot_patterns" in visualizations:
                fig = go.Figure(json.loads(visualizations["shot_patterns"]))
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Serve Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Overall serve analysis
            if "serve_analysis" in visualizations:
                fig = go.Figure(json.loads(visualizations["serve_analysis"]))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Player-specific serve analysis
            player_id = st.selectbox(
                "Select Player",
                options=[k.split("_")[1] for k in visualizations.keys() if k.startswith("player_") and k.endswith("_serve_analysis")],
                key="serve_player"
            )
            if player_id and f"player_{player_id}_serve_analysis" in visualizations:
                fig = go.Figure(json.loads(visualizations[f"player_{player_id}_serve_analysis"]))
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Dink Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Overall dink patterns
            if "dink_patterns" in visualizations:
                fig = go.Figure(json.loads(visualizations["dink_patterns"]))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Player-specific dink patterns
            player_id = st.selectbox(
                "Select Player",
                options=[k.split("_")[1] for k in visualizations.keys() if k.startswith("player_") and k.endswith("_dink_patterns")],
                key="dink_player"
            )
            if player_id and f"player_{player_id}_dink_patterns" in visualizations:
                fig = go.Figure(json.loads(visualizations[f"player_{player_id}_dink_patterns"]))
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Movement Analysis")
        if "player_movement" in visualizations:
            fig = go.Figure(json.loads(visualizations["player_movement"]))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Rally Analysis")
        if "rally_analysis" in visualizations:
            fig = go.Figure(json.loads(visualizations["rally_analysis"]))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("Performance Trends")
        if "performance_trends" in visualizations:
            fig = go.Figure(json.loads(visualizations["performance_trends"]))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.subheader("Combined View")
        # Display all visualizations in a grid
        cols = st.columns(2)
        for i, (viz_name, viz_json) in enumerate(visualizations.items()):
            if not viz_name.startswith("player_"):  # Only show overall visualizations
                with cols[i % 2]:
                    fig = go.Figure(json.loads(viz_json))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Add player-specific analysis section
        st.subheader("Player-Specific Analysis")
        player_id = st.selectbox(
            "Select Player",
            options=[k.split("_")[1] for k in visualizations.keys() if k.startswith("player_") and k.endswith("_serve_analysis")],
            key="combined_player"
        )
        if player_id:
            col1, col2, col3 = st.columns(3)
            with col1:
                if f"player_{player_id}_serve_analysis" in visualizations:
                    fig = go.Figure(json.loads(visualizations[f"player_{player_id}_serve_analysis"]))
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if f"player_{player_id}_dink_patterns" in visualizations:
                    fig = go.Figure(json.loads(visualizations[f"player_{player_id}_dink_patterns"]))
                    st.plotly_chart(fig, use_container_width=True)
            with col3:
                if f"player_{player_id}_shot_sequences" in visualizations:
                    fig = go.Figure(json.loads(visualizations[f"player_{player_id}_shot_sequences"]))
                    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit dashboard."""
    # Initialize session state
    if "token" not in st.session_state:
        st.session_state["token"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "page" not in st.session_state:
        st.session_state["page"] = "login"
    
    # Show login/register page if not authenticated
    if not st.session_state["token"]:
        if st.session_state["page"] == "login":
            login()
            if st.button("New user? Register here"):
                st.session_state["page"] = "register"
                st.experimental_rerun()
        else:
            register()
            if st.button("Already have an account? Login here"):
                st.session_state["page"] = "login"
                st.experimental_rerun()
        return
    
    # Show main dashboard
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["token"] = None
        st.session_state["username"] = None
        st.session_state["page"] = "login"
        st.experimental_rerun()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Strategy Analysis", "Team Analysis", "Performance Metrics", "Advanced Analysis"]
    )
    
    # Load user's games
    games = get_user_games()
    
    if page == "Strategy Analysis":
        # Load sample data for demonstration
        data = load_data()
        
        # Analyze strategy
        analysis = analyze_strategy(data)
        
        # Get visualizations
        visualizations = get_visualizations(analysis)
        
        # Show analysis
        show_strategy_analysis(analysis, visualizations)
        
    elif page == "Team Analysis":
        # Load sample team data
        team_data = {
            "team_id": "1",
            "players": [
                {"id": "1", "name": "Player 1"},
                {"id": "2", "name": "Player 2"}
            ],
            "games": games
        }
        
        # Show team analysis
        show_team_analysis(team_data)
        
    elif page == "Performance Metrics":
        # Show performance metrics
        show_performance_metrics(games)
        
    else:  # Advanced Analysis
        st.header("Advanced Analysis")
        
        # Game selection
        if games:
            selected_game = st.selectbox(
                "Select Game",
                options=games,
                format_func=lambda x: f"{x['date']} - {x['location']}"
            )
            
            if selected_game:
                # Get advanced analysis for selected game
                advanced_analysis = get_game_advanced_analysis(selected_game["id"])
                
                if advanced_analysis:
                    show_advanced_analysis(selected_game, advanced_analysis)
                else:
                    st.warning("No advanced analysis available for this game")
        else:
            st.info("No games available for analysis")

if __name__ == "__main__":
    main() 