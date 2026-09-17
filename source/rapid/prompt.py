##
# Goal prompts
##


goal_env_prompts = {
    "isaaclab_cabinet_drawer_open": "to open the drawer",
    "isaaclab_button_push": "to push the button down",
    "isaaclab_soccer_put_ball_in_goal": "to move the soccer ball into the goal",
    "isaaclab_sweep_into_cube_in_hole": "to sweep the cube into the hole in the table",
    "isaaclab_window_open": "to slide the window open as much as possible",
}


##
# Two stage analysis
##


gemini_free_query_prompt1 =  """
Consider the following two images:
Image 1:
"""

gemini_free_query_prompt2 = """
Image 2:
"""

gemini_free_query_env_prompts = {}
gemini_free_query_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

for env_name, prompt in goal_env_prompts.items():
    gemini_free_query_env_prompts[env_name] = gemini_free_query_template.format(prompt)
    
gemini_summary_env_prompts = {}

gemini_summary_template = """
Based on the text below to the questions:
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
{}

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
""" 

for env_name, prompt in goal_env_prompts.items():
    gemini_summary_env_prompts[env_name] = gemini_summary_template.format(prompt, "{}")


##
# One-query prompt
##


openrouter_single_query_combined_template = """
Consider the following two images:
Image 1: [Image 1]
Image 2: [Image 2]

1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?

Please answer the three questions above. After you finish the analysis,
on a new line write exactly the final label line in this format (and nothing else
on that line):

LABEL: <0 or 1 or -1>

- 0 means the goal is better achieved in Image 1.
- 1 means the goal is better achieved in Image 2.
- -1 means unsure or no difference.

You may include explanatory text BEFORE the LABEL line, but the LABEL line
must appear exactly as shown and must be on its own line so it can be parsed.
"""

openrouter_single_query_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    openrouter_single_query_env_prompts[env_name] = openrouter_single_query_combined_template.format(prompt)

