import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

"""
Sarashinaモデルを使用したGradioチャットボット
Hugging Face Transformersライブラリを使用してローカルでモデルを実行
"""

# モデルとトークナイザーの初期化
MODEL_NAME = "sbintuitions/sarashina2.2-3b-instruct-v0.1"

print("モデルを読み込み中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)
print("モデルの読み込みが完了しました。")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    チャットボットの応答を生成する関数
    Gradio ChatInterfaceの標準形式に対応
    """
    try:
        # パラメータを適切な型に変換（API呼び出し時の文字列対策）
        try:
            max_tokens = int(max_tokens) if max_tokens is not None else 512
            # max_tokensが0以下の場合は512に設定
            if max_tokens <= 0:
                max_tokens = 1
        except (ValueError, TypeError):
            max_tokens = 512
            
        try:
            temperature = float(temperature) if temperature is not None else 0.7
            # temperatureが0以下の場合は0.7に設定
            if temperature <= 0:
                temperature = 0.7
        except (ValueError, TypeError):
            temperature = 0.7
            
        try:
            top_p = float(top_p) if top_p is not None else 0.95
            # top_pが0以下の場合は0.95に設定
            if top_p <= 0:
                top_p = 0.95
        except (ValueError, TypeError):
            top_p = 0.95
        
        # システムメッセージと会話履歴を含むプロンプトを構築
        conversation = ""
        if system_message and system_message.strip():
            conversation += f"システム: {system_message}\n"
        
        # 会話履歴を追加
        for user_msg, bot_msg in history:
            if user_msg:
                conversation += f"ユーザー: {user_msg}\n"
            if bot_msg:
                conversation += f"アシスタント: {bot_msg}\n"
        
        # 現在のメッセージを追加
        conversation += f"ユーザー: {message}\nアシスタント: "
        
        # トークン化
        inputs = tokenizer.encode(conversation, return_tensors="pt")
        
        # GPU使用時はCUDAに移動
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # 応答生成（ストリーミング対応）
        response = ""
        with torch.no_grad():
            # 一度に生成してからストリーミング風に出力
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # 生成されたテキストをデコード
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 応答部分のみを抽出
        full_response = generated[len(conversation):].strip()
        
        # 不要な部分を除去
        if "ユーザー:" in full_response:
            full_response = full_response.split("ユーザー:")[0].strip()
        
        return full_response
            
            
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

"""
Gradio ChatInterfaceを使用したシンプルなチャットボット
カスタマイズ可能なパラメータを含む
"""
demo = gr.ChatInterface(
    respond,
    title="🤖 Sarashina Chatbot",
    description="Sarashina2.2-3b-instruct モデルを使用した日本語チャットボットです。",
    additional_inputs=[
        gr.Textbox(
            value="あなたは親切で知識豊富な日本語アシスタントです。ユーザーの質問に丁寧に答えてください。", 
            label="システムメッセージ",
            lines=3
        ),
        gr.Slider(
            minimum=1, 
            maximum=1024, 
            value=512, 
            step=1, 
            label="最大新規トークン数"
        ),
        gr.Slider(
            minimum=0.1, 
            maximum=2.0, 
            value=0.7, 
            step=0.1, 
            label="Temperature (創造性)"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (多様性制御)",
        ),
    ],
    theme=gr.themes.Soft(),
    examples=[
        ["こんにちは！今日はどんなことを話しましょうか？"],
        ["日本の文化について教えてください。"],
        ["簡単なレシピを教えてもらえますか？"],
        ["プログラミングについて質問があります。"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,  # API documentation を表示
        debug=True
    )