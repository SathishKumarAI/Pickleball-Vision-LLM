import pytest
from pickleball_vision.utils.colors import (
    Colors,
    colorize,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_debug
)

def test_colorize():
    """Test colorize function with different colors."""
    text = "Test message"
    
    # Test all color methods
    colored_text = colorize(text, Colors.BLUE)
    assert text in colored_text
    assert "\033[" in colored_text  # ANSI color code
    
    # Test with different colors
    for color in Colors.__dict__.values():
        if isinstance(color, str) and color.startswith("\033["):
            result = colorize(text, color)
            assert text in result
            assert color in result

def test_print_functions(capsys):
    """Test all print functions."""
    test_message = "Test message"
    
    # Test success message
    print_success(test_message)
    captured = capsys.readouterr()
    assert test_message in captured.out
    assert Colors.GREEN in captured.out
    
    # Test error message
    print_error(test_message)
    captured = capsys.readouterr()
    assert test_message in captured.out
    assert Colors.RED in captured.out
    
    # Test warning message
    print_warning(test_message)
    captured = capsys.readouterr()
    assert test_message in captured.out
    assert Colors.YELLOW in captured.out
    
    # Test info message
    print_info(test_message)
    captured = capsys.readouterr()
    assert test_message in captured.out
    assert Colors.BLUE in captured.out
    
    # Test debug message
    print_debug(test_message)
    captured = capsys.readouterr()
    assert test_message in captured.out
    assert Colors.CYAN in captured.out

def test_colorize_with_custom_format():
    """Test colorize with custom format."""
    text = "Test message"
    custom_format = "Message: {}"
    
    result = colorize(text, Colors.BLUE, custom_format)
    expected = f"{Colors.BLUE}Message: {text}{Colors.RESET}"
    assert result == expected

def test_colorize_with_empty_text():
    """Test colorize with empty text."""
    result = colorize("", Colors.BLUE)
    assert result == ""

def test_print_functions_with_empty_message(capsys):
    """Test print functions with empty message."""
    print_success("")
    captured = capsys.readouterr()
    assert captured.out == "\n"
    
    print_error("")
    captured = capsys.readouterr()
    assert captured.out == "\n"
    
    print_warning("")
    captured = capsys.readouterr()
    assert captured.out == "\n"
    
    print_info("")
    captured = capsys.readouterr()
    assert captured.out == "\n"
    
    print_debug("")
    captured = capsys.readouterr()
    assert captured.out == "\n"

def test_colorize_with_special_characters():
    """Test colorize with special characters."""
    text = "Test\nmessage\twith\r\nspecial chars"
    result = colorize(text, Colors.BLUE)
    assert text in result
    assert Colors.BLUE in result 