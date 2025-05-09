from typing import Dict

"""
prompt_templates.py

Modular prompt templates for LLM + Computer Vision tasks in Pickleball Vision Model.
Author: Sathish Kumar
Email: sathishkumar786.ml@gmail.com
"""



class PromptTemplates:
    """Centralized repository for prompt templates used across the pipeline."""

    def __init__(self):
        self.templates = {
            "frame_captioning": (
                "You're a professional sports video analyst. "
                "Describe this image from a Pickleball match in one concise sentence:"
            ),
            "pose_feedback": (
                "Given the following keypoints for a player during a pickleball match, "
                "provide biomechanical feedback on posture, movement quality, and balance:\n\n{keypoints}"
            ),
            "clip_interpretation": (
                "You are an AI coach. Use the following scene description from a Pickleball clip:\n\n"
                "\"{caption}\"\n\n"
                "Give tactical or coaching advice based on this action."
            ),
            "semantic_qa": (
                "Use the following video segment and description to answer the question:\n\n"
                "Caption: \"{caption}\"\n"
                "Question: \"{question}\"\n"
                "Answer:"
            ),
            "game_summary": (
                "Analyze this sequence of frames and generate a high-level summary of the Pickleball match:\n\n"
                "{captions}\n\n"
                "Summarize like a commentator:"
            ),
        }

    def get(self, name: str, **kwargs) -> str:
        """Get a formatted prompt template with inserted values."""
        if name not in self.templates:
            raise ValueError(f"Prompt template '{name}' not found.")

        prompt = self.templates[name]
        return prompt.format(**kwargs)


# Example usage
if __name__ == "__main__":
    prompts = PromptTemplates()
    caption_prompt = prompts.get("frame_captioning")
    print("[Frame Caption Prompt]:\n", caption_prompt)

    qa_prompt = prompts.get("semantic_qa", caption="Player dives to return a serve.", question="Was this an offensive or defensive play?")
    print("\n[Semantic QA Prompt]:\n", qa_prompt)