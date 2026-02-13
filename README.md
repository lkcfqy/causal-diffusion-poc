# 🌟 Causal Diffusion POC (Proof of Concept) 🎨✨

Welcome to the **Causal Diffusion POC**! 🚀 This is an exciting experimental project where **Causal Inference** meets **Diffusion Models**! 🧠💭

In this repository, we play with colored MNIST digits (specifically our friends 3️⃣ and 7️⃣, dressed in 🔴 red and 🟢 green) to explore how structural causal models can give us better interpretability and control over the image generation process! 🪄🖼️

---

## ✨ What is this about? 🧐

This Proof of Concept (POC) compares a standard **Baseline Diffusion Model** with a **Structural Causal Diffusion (SCD) Model**! We are testing two super exciting hypotheses:

* 🛡️ **H1: Robustness**: Can our causal model generate better, more robust images when dealing with interventions or structural shifts? (Spoiler: We test it!)
* 🔄 **H2: Counterfactuals**: *"What if this green 3 was actually a 7?"* 🤔 We intervene on specific causal variables (like digit class or color) to generate mind-bending counterfactual images! 🌈

---

## 📂 Project Structure 🗺️

Here's a quick tour of our cozy codebase:

* 📁 `src/` 💻: The brain of the operation!
    * 📄 `create_dataset.py`: Whips up our special colored MNIST dataset. 🎨
    * 📄 `diffusion.py`: The core diffusion math and magic! 🔢
    * 📄 `models.py`: Neural network architectures for our models. 🕸️
    * 📄 `train.py`: Time to hit the gym! Trains our Baseline and SCD models. 🏋️‍♀️
    * 📄 `evaluate.py`: Puts the models to the test to generate our H1 and H2 results. 📊
* 📄 `config.yaml` ⚙️: All the secret dials and configuration parameters.
* 📁 `saved_models/` 💾: Home to our pre-trained beauties (`baseline_model.pt` & `scd_model.pt`).
* 📁 `results/` 🏆: Where the magic is stored! Contains data samples, H1 robustness outputs, and H2 counterfactual generations. 🖼️✨

---

## 🛠️ Getting Started 🚀

Ready to run some causal magic? Follow these simple steps!

### 1. Set up the Environment 🌱
We use Conda to keep things clean and tidy! 🧹
```bash
conda env create -f environment.yml
conda activate causal-diffusion

```

### 2. Create the Dataset 🎨

Let's paint some 3s and 7s!

```bash
python src/create_dataset.py

```

### 3. Train the Models 🏋️‍♂️

*Grab a coffee ☕, this might take a minute!*

```bash
python src/train.py

```

### 4. Evaluate & Generate Results 🪄

Time to see the counterfactual magic and robustness tests!

```bash
python src/evaluate.py

```

*(Check the `results/` folder to see your gorgeous generated images!)* 🤩

---

## 🖼️ Sneak Peek at Results 👀

* **Data Samples**: Check out `results/data_samples/` to see our training data (e.g., green 7s, red 3s). 🟢🔴
* **Counterfactuals (H2)**: Open `results/h2_counterfactuals/` to see the models seamlessly turning a Red 3 into a Red 7 without losing the color style! 🤯✨

---

*Built with ❤️ and 🧠 for pushing the boundaries of generative AI! Happy experimenting!* 🎉
