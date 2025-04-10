import gradio as gr
from qa_pipeline import answer_user_question

def qa_interface(question):
    print(f"💬 問題: {question}")
    answer = answer_user_question(question)
    return answer

#Gradio 介面
demo = gr.Interface(
    fn=qa_interface,
    inputs=gr.Textbox(lines=5, placeholder="請入你的問題"),
    outputs="text",
    title="商品情緒分析與摘要",
    description="根據用戶評論，分析商品情緒並生成評論總結，支持問題提問"
)

if __name__ == "__main__":
    demo.launch()

