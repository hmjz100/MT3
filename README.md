---
title: MT3
emoji: 🎼
colorFrom: purple
colorTo: green
sdk: gradio
app_file: app.py
pinned: true
duplicated_from: oniati/mrt
---
# MT3
多任务多音轨音乐转录  
HuggingFace打不开？试试用 [hmjz100/Super-Nginx](https://github.com/hmjz100/Super-Nginx) 吧！

---
# Configuration
`title`: _string_  
Display title for the Space
`emoji`: _string_  
Space emoji (emoji-only character allowed)
`colorFrom`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)
`colorTo`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)
`sdk`: _string_  
Can be either `gradio` or `streamlit`
`sdk_version` : _string_  
Only applicable for `streamlit` SDK.  
See [doc](https://hf.co/docs/hub/spaces) for more info on supported versions.

`app_file`: _string_  
Path to your main application file (which contains either `gradio` or `streamlit` Python code).  
Path is relative to the root of the repository.

`pinned`: _boolean_  
Whether the Space stays on top of your list.
