---
title: Data Science - Facial Landmark Emotion Classification - University of Lille
author:
  name: Pierre Lague & François Muller (@franzele21)
  link: 
date: 2023-12-19 09:45:00 +0800
categories: [Studies, U-Lille, Python]
tags: [Python, ML]
math: true
mermaid: true
image:
  src: '/assets/posts/SD-facial-emotion/header.jpeg'
  width: 800
  height: 600
---

# FLEC - Facial Landmark Emotion Classification Project

Welcome to the **Facial Landmark Emotion Classification (FLEC)** project! Our goal is to develop a robust classifier capable of recognizing six facial expressions using the 68 facial landmarks provided in .csv files.

## Project Overview :mag:

### Step 1: Characterizing Expressions 🎯

In this crucial step, we aim to predict the `emotion` column in the `emotion.csv` file based on the facial points available in the `SXXX/omlands.csv` files. Our approaches include:

1. Utilizing the coordinates of facial points.
2. Analyzing the movement of points between neutral and apex emotion images.

To enhance accuracy, we'll experiment with face alignment techniques, exploring both the use of raw points and a common frame of reference.

### Step 2: Handling Imbalance 🙌

Given the highly imbalanced dataset, our second phase focuses on creating a balanced dataset. We'll assess the impact on results compared to the initial configuration, ensuring more reliable and unbiased model training.

### Step 3: Study of Occlusions and Noises 🕶️

Understanding the impact of occlusions and noises on facial landmarks is crucial for real-world applications. 

#### Step 3.1: Creating Occlusions and Noises 👥

We'll simulate various occlusions and noises, starting from small regions (e.g., eyes, eyebrows) to larger occlusions. This step aims to evaluate model robustness under different alteration scenarios.

#### Step 3.2: Evaluating Robustness 📊

Our evaluation will provide insights into how well our learning techniques handle occlusions and noises. Quantification and appropriate measurements will guide our assessment of robustness.

## Project Report 🛠️

Following is the official report for the project.
<html>
  <body>
    <iframe src="/assets/posts/SD-facial-emotion/LAGUE_MULLER_REPORT.pdf" width="100%" height="800px">
    </iframe>
  </body>
</html>


## Contributing 🤝

Contributions are highly encouraged! If you have suggestions, improvements, or feature requests, feel free to open an issue or create a pull request on our [public repository](https://github.com/Jakcrimson/pjeb_twitter_sentiment_analysis).

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed by Pierre LAGUE and François MULLER (@franzele21) at the University of Lille, France. 🚀📊

Let's make facial emotion classification more accurate and reliable together! 🌟👥