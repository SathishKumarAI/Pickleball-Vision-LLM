# Frame Analysis Prompts

This document contains prompts for various frame analysis tasks in the Pickleball Vision project.

## Frame Quality Assessment

```prompt
Analyze the given frame for quality metrics:
- Assess brightness (target range: 0.3-0.8 normalized)
- Check contrast (minimum std dev: 0.15)
- Detect blur (Laplacian variance threshold: 100)
- Evaluate color balance

Input: [frame data]
Expected Output: Quality metrics JSON with pass/fail status
```

## Motion Detection

```prompt
Detect and analyze motion between consecutive frames:
- Calculate optical flow using Farneback method
- Identify regions of significant movement
- Track ball trajectory patterns
- Filter out camera motion

Input: [consecutive frames]
Expected Output: Motion vectors and ball movement probability
```

## Frame Preprocessing

```prompt
Preprocess the input frame for optimal ball detection:
- Resize to standard dimensions (1280x720)
- Apply color space conversion (BGR to RGB)
- Normalize pixel values
- Enhance contrast if needed

Input: [raw frame]
Expected Output: Preprocessed frame ready for detection
```

## Usage Instructions

1. Copy the relevant prompt template
2. Replace [input placeholders] with actual data
3. Run through the appropriate processing pipeline
4. Validate outputs against expected formats
5. Adjust parameters based on results

## Best Practices

- Always validate input frame dimensions
- Check for color space consistency
- Log quality metrics for analysis
- Save problematic frames for review
- Document parameter adjustments 