"""Test suite for pickleball analyzers."""
import pytest
import numpy as np
import torch
import asyncio
from datetime import datetime, timedelta
from src.pickleball_vision.analytics.gpu_analyzer import GPUAnalyzer
from src.pickleball_vision.analytics.ml_analyzer import MLAnalyzer
from src.pickleball_vision.analytics.stream_analyzer import StreamAnalyzer

# Extended test data
SAMPLE_SHOTS = [
    {
        "placement_x": 0.5,
        "placement_y": 0.3,
        "speed": 30.0,
        "spin": 1000.0,
        "effectiveness_score": 0.8,
        "player_position_x": 0.2,
        "player_position_y": 0.4,
        "opponent_position_x": 0.8,
        "success": True,
        "shot_type": "drive",
        "player_id": "player1",
        "rally_id": "rally1"
    },
    {
        "placement_x": 0.7,
        "placement_y": 0.5,
        "speed": 35.0,
        "spin": 1200.0,
        "effectiveness_score": 0.9,
        "player_position_x": 0.3,
        "player_position_y": 0.5,
        "opponent_position_x": 0.7,
        "success": True,
        "shot_type": "drop",
        "player_id": "player1",
        "rally_id": "rally1"
    },
    {
        "placement_x": 0.3,
        "placement_y": 0.6,
        "speed": 25.0,
        "spin": 800.0,
        "effectiveness_score": 0.7,
        "player_position_x": 0.6,
        "player_position_y": 0.3,
        "opponent_position_x": 0.4,
        "success": False,
        "shot_type": "lob",
        "player_id": "player2",
        "rally_id": "rally2"
    }
]

SAMPLE_POSITIONS = [
    {
        "x": 0.2,
        "y": 0.4,
        "speed": 5.0,
        "timestamp": datetime.now().timestamp(),
        "player_id": "player1",
        "team": "team1"
    },
    {
        "x": 0.3,
        "y": 0.5,
        "speed": 6.0,
        "timestamp": datetime.now().timestamp() + 1,
        "player_id": "player1",
        "team": "team1"
    },
    {
        "x": 0.7,
        "y": 0.3,
        "speed": 4.5,
        "timestamp": datetime.now().timestamp() + 2,
        "player_id": "player2",
        "team": "team2"
    }
]

SAMPLE_RALLIES = [
    {
        "id": "rally1",
        "duration": 10.0,
        "winner_team": "team1",
        "shot_count": 5,
        "start_time": datetime.now().timestamp(),
        "end_time": datetime.now().timestamp() + 10,
        "players": ["player1", "player2"]
    },
    {
        "id": "rally2",
        "duration": 15.0,
        "winner_team": "team2",
        "shot_count": 7,
        "start_time": datetime.now().timestamp() + 15,
        "end_time": datetime.now().timestamp() + 30,
        "players": ["player1", "player2"]
    }
]

# Performance test data
PERFORMANCE_SHOTS = [
    {
        "placement_x": np.random.random(),
        "placement_y": np.random.random(),
        "speed": np.random.uniform(20, 40),
        "spin": np.random.uniform(500, 1500),
        "effectiveness_score": np.random.random(),
        "player_position_x": np.random.random(),
        "player_position_y": np.random.random(),
        "opponent_position_x": np.random.random(),
        "success": np.random.choice([True, False]),
        "shot_type": np.random.choice(["drive", "drop", "lob", "smash"]),
        "player_id": f"player{i % 4 + 1}",
        "rally_id": f"rally{i // 10 + 1}"
    }
    for i in range(1000)
]

@pytest.fixture
def gpu_analyzer():
    """Create GPU analyzer instance."""
    return GPUAnalyzer()

@pytest.fixture
def ml_analyzer():
    """Create ML analyzer instance."""
    return MLAnalyzer()

@pytest.fixture
def stream_analyzer():
    """Create stream analyzer instance."""
    return StreamAnalyzer()

def test_gpu_analyzer_shot_patterns(gpu_analyzer):
    """Test GPU analyzer shot pattern analysis."""
    result = gpu_analyzer.analyze_shot_patterns(SAMPLE_SHOTS)
    assert "patterns" in result
    assert "statistics" in result
    assert isinstance(result["patterns"], list)
    assert isinstance(result["statistics"], dict)
    
    # Test pattern details
    pattern = result["patterns"][0]
    assert "positions" in pattern
    assert "speeds" in pattern
    assert "spins" in pattern
    assert "effectiveness" in pattern
    
    # Test statistics
    stats = result["statistics"]
    assert "avg_speed" in stats
    assert "avg_spin" in stats
    assert "avg_effectiveness" in stats

def test_gpu_analyzer_player_movement(gpu_analyzer):
    """Test GPU analyzer player movement analysis."""
    result = gpu_analyzer.analyze_player_movement(SAMPLE_POSITIONS)
    assert "metrics" in result
    assert "patterns" in result
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["patterns"], list)
    
    # Test movement metrics
    metrics = result["metrics"]
    assert "velocities" in metrics
    assert "accelerations" in metrics
    assert "distances" in metrics
    assert "total_distance" in metrics
    
    # Test movement patterns
    pattern = result["patterns"][0]
    assert "type" in pattern
    assert "positions" in pattern
    assert "metrics" in pattern

def test_gpu_analyzer_rally_dynamics(gpu_analyzer):
    """Test GPU analyzer rally dynamics analysis."""
    result = gpu_analyzer.analyze_rally_dynamics(SAMPLE_RALLIES, SAMPLE_SHOTS)
    assert "metrics" in result
    assert "patterns" in result
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["patterns"], list)
    
    # Test rally metrics
    metrics = result["metrics"]
    assert "avg_duration" in metrics
    assert "avg_shots" in metrics
    assert "win_rate" in metrics
    assert "total_rallies" in metrics
    
    # Test rally patterns
    pattern = result["patterns"][0]
    assert "duration" in pattern
    assert "shot_count" in pattern
    assert "win_rate" in pattern
    assert "frequency" in pattern

def test_ml_analyzer_shot_predictor_training(ml_analyzer):
    """Test ML analyzer shot predictor training."""
    result = ml_analyzer.train_shot_predictor(SAMPLE_SHOTS)
    assert "train_losses" in result
    assert "val_losses" in result
    assert isinstance(result["train_losses"], list)
    assert isinstance(result["val_losses"], list)
    
    # Test training progress
    assert len(result["train_losses"]) > 0
    assert len(result["val_losses"]) > 0
    assert result["train_losses"][-1] < result["train_losses"][0]  # Loss should decrease

def test_ml_analyzer_shot_prediction(ml_analyzer):
    """Test ML analyzer shot prediction."""
    # Train the model first
    ml_analyzer.train_shot_predictor(SAMPLE_SHOTS)
    
    # Test prediction
    result = ml_analyzer.predict_shot_outcome(SAMPLE_SHOTS[0])
    assert "success_probability" in result
    assert isinstance(result["success_probability"], float)
    assert 0 <= result["success_probability"] <= 1
    
    # Test multiple predictions
    predictions = [
        ml_analyzer.predict_shot_outcome(shot)
        for shot in SAMPLE_SHOTS
    ]
    assert len(predictions) == len(SAMPLE_SHOTS)
    assert all(0 <= p["success_probability"] <= 1 for p in predictions)

def test_ml_analyzer_player_style(ml_analyzer):
    """Test ML analyzer player style analysis."""
    result = ml_analyzer.analyze_player_style(SAMPLE_SHOTS)
    assert "style_type" in result
    assert "metrics" in result
    assert isinstance(result["style_type"], str)
    assert isinstance(result["metrics"], dict)
    
    # Test style metrics
    metrics = result["metrics"]
    assert any(f"cluster_{i}" in metrics for i in range(3))
    for cluster in metrics.values():
        assert "avg_speed" in cluster
        assert "avg_spin" in cluster
        assert "success_rate" in cluster
        assert "frequency" in cluster

@pytest.mark.asyncio
async def test_stream_analyzer(stream_analyzer):
    """Test stream analyzer functionality."""
    # Test data source
    def data_source():
        return {
            "shots": SAMPLE_SHOTS,
            "positions": SAMPLE_POSITIONS,
            "rallies": SAMPLE_RALLIES
        }
    
    # Test callback
    callback_results = []
    def callback(result):
        callback_results.append(result)
    
    # Start stream
    stream_id = "test_stream"
    await stream_analyzer.start_stream(
        stream_id,
        data_source,
        {"use_gpu": True, "use_ml": True}
    )
    
    # Register callback
    stream_analyzer.register_callback(stream_id, callback)
    
    # Wait for some analysis cycles
    await asyncio.sleep(2)
    
    # Check stream status
    status = stream_analyzer.get_stream_status(stream_id)
    assert status["status"] == "running"
    assert "config" in status
    
    # Check active streams
    active_streams = stream_analyzer.get_active_streams()
    assert stream_id in active_streams
    
    # Stop stream
    await stream_analyzer.stop_stream(stream_id)
    
    # Verify callback was called
    assert len(callback_results) > 0
    assert "results" in callback_results[0]
    assert "stream_id" in callback_results[0]
    
    # Test multiple streams
    stream_ids = [f"stream_{i}" for i in range(3)]
    for sid in stream_ids:
        await stream_analyzer.start_stream(sid, data_source, {"use_gpu": True, "use_ml": True})
    
    active_streams = stream_analyzer.get_active_streams()
    assert all(sid in active_streams for sid in stream_ids)
    
    for sid in stream_ids:
        await stream_analyzer.stop_stream(sid)

def test_error_handling():
    """Test error handling in analyzers."""
    gpu_analyzer = GPUAnalyzer()
    ml_analyzer = MLAnalyzer()
    
    # Test invalid shot data
    with pytest.raises(ValueError):
        gpu_analyzer.analyze_shot_patterns([{"invalid": "data"}])
    
    # Test invalid position data
    with pytest.raises(ValueError):
        gpu_analyzer.analyze_player_movement([{"invalid": "data"}])
    
    # Test invalid rally data
    with pytest.raises(ValueError):
        gpu_analyzer.analyze_rally_dynamics([{"invalid": "data"}], [])
    
    # Test prediction without training
    with pytest.raises(ValueError):
        ml_analyzer.predict_shot_outcome({"invalid": "data"})
    
    # Test empty data
    with pytest.raises(ValueError):
        gpu_analyzer.analyze_shot_patterns([])
    
    # Test missing required fields
    with pytest.raises(ValueError):
        gpu_analyzer.analyze_player_movement([{"x": 0.5}])  # Missing y and other fields

def test_data_validation():
    """Test data validation in analyzers."""
    gpu_analyzer = GPUAnalyzer()
    ml_analyzer = MLAnalyzer()
    
    # Test missing required fields
    invalid_shots = [{"placement_x": 0.5}]  # Missing other required fields
    assert not gpu_analyzer._validate_data(
        invalid_shots,
        ["placement_x", "placement_y", "speed", "spin", "effectiveness_score"]
    )
    
    # Test data preparation
    prepared_data = gpu_analyzer._prepare_data(invalid_shots)
    assert len(prepared_data) == 1
    assert "placement_x" in prepared_data[0]
    
    # Test data cleaning
    dirty_data = [
        {"placement_x": 0.5, "placement_y": None, "speed": 30.0},
        {"placement_x": 0.7, "placement_y": 0.3, "speed": None}
    ]
    cleaned_data = gpu_analyzer._prepare_data(dirty_data)
    assert len(cleaned_data) == 2
    assert all(k in cleaned_data[0] for k in ["placement_x", "speed"])
    assert all(k in cleaned_data[1] for k in ["placement_x", "placement_y"])

def test_response_format():
    """Test response format standardization."""
    gpu_analyzer = GPUAnalyzer()
    
    # Test response creation
    data = {"test": "data"}
    response = gpu_analyzer._create_response(data)
    assert "test" in response
    assert "timestamp" in response
    assert isinstance(response["timestamp"], str)
    
    # Test custom timestamp
    custom_time = datetime.now()
    response = gpu_analyzer._create_response(data, custom_time)
    assert response["timestamp"] == custom_time.isoformat()

@pytest.mark.performance
def test_performance():
    """Test performance with large datasets."""
    gpu_analyzer = GPUAnalyzer()
    ml_analyzer = MLAnalyzer()
    
    # Test GPU analyzer performance
    start_time = datetime.now()
    result = gpu_analyzer.analyze_shot_patterns(PERFORMANCE_SHOTS)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    assert processing_time < 5.0  # Should process 1000 shots in under 5 seconds
    assert len(result["patterns"]) > 0
    
    # Test ML analyzer performance
    start_time = datetime.now()
    ml_analyzer.train_shot_predictor(PERFORMANCE_SHOTS[:100])  # Use subset for training
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    assert training_time < 10.0  # Should train in under 10 seconds
    
    # Test prediction performance
    start_time = datetime.now()
    predictions = [
        ml_analyzer.predict_shot_outcome(shot)
        for shot in PERFORMANCE_SHOTS[:100]
    ]
    end_time = datetime.now()
    prediction_time = (end_time - start_time).total_seconds()
    
    assert prediction_time < 5.0  # Should predict 100 shots in under 5 seconds
    assert len(predictions) == 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 