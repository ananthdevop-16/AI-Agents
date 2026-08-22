import os
import tempfile
import re

import streamlit as st
from langchain_core.messages import AIMessage

from shopping_agent import agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want — I will search, rank, and help you order.")

# ---------------------------------------------------------------------------
# Helper: safe extraction
# ---------------------------------------------------------------------------
def get_final_answer(messages):
    """
    Extract clean final AI response from LangGraph messages
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content

            # dict output (thinking leak)
            if isinstance(content, dict):
                if "thinking" in content:
                    continue
                content = content.get("text", "")

            # list output (multimodal)
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )

            content = str(content)

            # remove backticks + thinking blocks
            content = content.replace("`", "")
            content = re.sub(r"\{.*?thinking.*?\}", "", content, flags=re.DOTALL)

            return content.strip()

    return "No response generated."

# ---------------------------------------------------------------------------
# Sidebar — image search
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Shop by Image")
    uploaded_file = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if uploaded_file and st.button("Find similar products"):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        prompt = (
            "I uploaded a product image. "
            
        )

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        st.rerun()

# ---------------------------------------------------------------------------
# Init session
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------
def run_agent():
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": st.session_state.messages}
            )

            response = get_final_answer(result["messages"])

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# ---------------------------------------------------------------------------
# Handle pending image request
# ---------------------------------------------------------------------------
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    run_agent()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("e.g. I want organic honey under $15"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    run_agent()

# import os
# import tempfile

# import streamlit as st
# from langchain_core.messages import AIMessage
# from shopping_agent import agent

# # ---------------------------------------------------------------------------
# # Page config
# # ---------------------------------------------------------------------------
# st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")

# st.title("🛒 AI Shopping Assistant")
# st.caption("Tell me what you want — I'll search, rate, and order the best match for you.")

# # ---------------------------------------------------------------------------
# # Sidebar — shop by image
# # ---------------------------------------------------------------------------
# with st.sidebar:
#     st.header("Shop by Image")
#     st.caption("Upload a photo of a product and I'll find similar items in our store.")

#     uploaded_file = st.file_uploader(
#         "Upload product image", type=["jpg", "jpeg", "png", "webp"]
#     )

#     if uploaded_file:
#         st.image(uploaded_file, use_container_width=True)

#     if uploaded_file and st.button("Find similar products", use_container_width=True):
#         suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp.write(uploaded_file.getvalue())
#             image_path = tmp.name

#         prompt = f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}"
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.session_state.pending_image = uploaded_file.name
#         st.rerun()

# # ---------------------------------------------------------------------------
# # Chat state
# # ---------------------------------------------------------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Render history — show a friendlier label for image-search messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         if msg["role"] == "user" and msg["content"].startswith("I uploaded a product image"):
#             filename = msg["content"].split("Image path:")[-1].strip()
#             st.markdown(f"Searching by image: **{os.path.basename(filename)}**")
#         else:
#             st.markdown(msg["content"].replace("$", r"\$"))

# # ---------------------------------------------------------------------------
# # Run agent if there's an unprocessed message (image upload triggers this)
# # ---------------------------------------------------------------------------
# if (
#     st.session_state.messages
#     and st.session_state.messages[-1]["role"] == "user"
#     and "pending_image" in st.session_state
# ):
#     with st.chat_message("assistant"):
#         with st.spinner("Analyzing image and searching…"):
#             result = agent.invoke({"messages": st.session_state.messages})
#             response = result["messages"][-1].content.replace("`", "")
#         st.markdown(response.replace("$", r"\$"))

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     del st.session_state.pending_image
#     st.rerun()

# # ---------------------------------------------------------------------------
# # Text input
# # ---------------------------------------------------------------------------
# if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking…"):
#             result = agent.invoke({"messages": st.session_state.messages})
#             #response = result["messages"][-1].content.replace("`", "")

#             messages = result["messages"]
#             final_text = None

#             for msg in reversed(messages):
#                 if isinstance(msg, AIMessage):
#                     content = msg.content

#                     # if it's a dict-like thinking output → skip
#                     if isinstance(content, dict) and "thinking" in content:
#                         continue

#                     # if it's list → flatten
#                     if isinstance(content, list):
#                         content = " ".join(
#                             part.get("text", "") if isinstance(part, dict) else str(part)
#                             for part in content
#                         )

#                     final_text = content
#                     break

#             response=(final_text)

#             ###
#             #msg = result["messages"][-1]

#             #content = msg.content

#             # if isinstance(content, list):
#             #     content = " ".join(
#             #         part["text"] if isinstance(part, dict) and "text" in part else str(part)
#             #         for part in content
#             #     )

#             # response = content.replace("`", "")


#         st.markdown(response.replace("$", r"\$"))

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.rerun()
