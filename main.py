import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import asyncio

# 既存のイベントループをチェックしてなければ作る
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# タイトルとイントロの設定
st.title("Conditional GAN MNIST画像生成")
st.write("保存されたGANモデルを使って、選択された数字の画像を生成します。")

# ハイパーパラメータ（元のコードと一致させる）
latent_dim = 10
n_classes = 10

# 使用デバイスの設定（GPUがあればGPU、なければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"使用デバイス: {device}")

# Generator の定義（元のコードと一致させる）
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4  # 28//4 = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 7x7 -> 14x14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 14x14 -> 28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()  # 出力は [-1, 1]
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)               # (batch_size, n_classes)
        gen_input = torch.cat((noise, label_input), -1)      # (batch_size, latent_dim+n_classes)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator の定義（元のコードと一致させる）
class Discriminator(nn.Module):
    def __init__(self, n_classes, img_size=28):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(n_classes + img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 出力は [0,1] の確率
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)                # (batch_size, 28*28)
        label_input = self.label_embedding(labels)          # (batch_size, n_classes)
        d_in = torch.cat((img_flat, label_input), -1)         # (batch_size, 28*28+n_classes)
        validity = self.model(d_in)
        return validity

# 画像生成関数の定義
def load_model():
    try:
        # モデルのロード
        generator = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
        return generator
    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生しました: {e}")
        return None

def generate_images(generator, device, latent_dim, target_label, n_images=5):
    """
    指定されたラベルに基づいて画像を生成する関数
    """
    if generator is None:
        return None
    
    # 対象のラベルを Tensor に変換
    label_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
    # n_images 枚生成するためラベルを繰り返す
    gen_labels = label_tensor.repeat(n_images)

    # ランダムノイズ生成
    z = torch.randn(n_images, latent_dim, device=device)

    # 推論モードで画像生成
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(z, gen_labels)
    
    # Generator の出力は Tanh により [-1,1] なので、[0,1] にスケーリング
    gen_imgs = (gen_imgs + 1) / 2
    
    return gen_imgs

# モデルをロード
generator = load_model()

if generator:
    # サイドバーでパラメータを設定
    st.sidebar.header("生成パラメータ")
    target_label = st.sidebar.selectbox("数字を選択", list(range(10)))
    n_images = st.sidebar.slider("生成する画像の数", 1, 10, 5)
    random_seed = st.sidebar.number_input("ランダムシード（任意）", value=-1)
    
    # 生成ボタン
    if st.sidebar.button("画像を生成"):
        if random_seed >= 0:
            torch.manual_seed(random_seed)
            
        # 画像を生成
        gen_imgs = generate_images(generator, device, latent_dim, target_label, n_images)
        
        if gen_imgs is not None:
            # 画像を表示
            st.subheader(f"数字 {target_label} の生成画像")
            
            # 画像を横に並べて表示
            fig, axs = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
            if n_images == 1:
                axs = [axs]
            
            for i, ax in enumerate(axs):
                img = gen_imgs[i].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f"イメージ {i+1}")
                ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 画像を個別にダウンロード可能に
            st.subheader("画像のダウンロード")
            for i in range(n_images):
                img = gen_imgs[i].cpu().squeeze().numpy()
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                
                # PILイメージをバイトに変換
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                
                # ダウンロードボタン
                st.download_button(
                    label=f"画像 {i+1} をダウンロード",
                    data=buf.getvalue(),
                    file_name=f"generated_{target_label}_{i}.png",
                    mime="image/png"
                )
else:
    st.error("モデルをロードできませんでした。モデルファイル 'GANgenerator.pth' が正しい場所にあるか確認してください。")

# フッター
st.markdown("---")
st.markdown("Conditional GAN を使った MNIST 数字生成アプリケーション")
