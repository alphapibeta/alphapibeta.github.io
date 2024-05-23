---
layout: page
title: Facial Detection and Recognition Project
description: Internship project featuring facial detection and recognition with age and gender estimation.
img: assets/img/facerecg.png
importance: 1
category: Wavelabs-Internship
---

<div class="container mt-4">
    <div class="row">
        <div class="col-lg-12">
            {% include figure.liquid path="assets/img/face_recg.gif" title="emYt+ dashboard" class="img-fluid rounded z-depth-1" %}
            <div class="caption text-center">
                face detection and recognition
            </div>
        </div>
    </div>
</div>

<div class="container mt-4">
    <div class="row">
        <div class="col-lg-12">
            <h2>Facial Detection and Recognition with Age and Gender Estimation</h2>
            <p>An in-house project designed to showcase our capabilities in security solutions to clients at Wavelabs.ai. This system integrates facial detection and recognition with age and gender estimation, capable of real-time inference, even on modest hardware.</p>

            <h3>Project Highlights</h3>
            <ul>
                <li>Developed a robust facial detection system using ResNet50, trained on a dataset of over 5000+ images.</li>
                <li>Implemented Histogram of Oriented Gradients (HOG) for face extraction achieving 40+ FPS with a resolution of 416x416.</li>
                <li>Utilized an ensemble approach with two separate models for age estimation (regression) and gender classification (classification).</li>
                <li>Achieved real-time inference on an i5 2.3 GHz processor with a frame rate of 8-10 FPS.</li>
                <li>Encoded facial features into a 300-dimensional vector for accurate face identification.</li>
            </ul>

            <h3>Tech Stack</h3>
            <ul>
                <li><strong>Computer Vision:</strong> OpenCV for image processing and face detection.</li>
                <li><strong>Machine Learning:</strong> Keras for building the deep learning models for age and gender estimation.</li>
            </ul>

            <h3>Additional Work</h3>
            <p>Alongside the facial recognition system, I also contributed to an image classification project:</p>
            <ul>
                <li>Collected and manually labeled 3000+ images from the internet for multiple base architectures.</li>
                <li>Applied object detection techniques to the dataset to create bounding boxes.</li>
            </ul>
        </div>
    </div>

</div>
