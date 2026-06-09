# Lesson 01: Intro to Video Understanding

## 1. Objectives

This lecture introduces the foundations of video understanding in computer vision.

Main objectives:

- Understand the key differences between image processing and video processing.
- Define the temporal dimension mathematically.
- Understand why adding time greatly increases tensor size and memory cost.
- Identify computational bottlenecks caused by video compression and frame extraction.
- Learn how to design data loading pipelines for 4D video tensors using common libraries.

---

## 2. What Is Video Understanding?

Video understanding means automatically extracting meaningful information from video data.

Unlike image analysis, video understanding must interpret two types of information:

- Spatial content: what appears in each frame.
- Temporal dynamics: how the scene changes across frames.

Example:

- Image analysis: “There is a door.”
- Video understanding: “The door is opening.”

The key difference is that video contains motion and time-dependent changes.

---

## 3. Video Processing Pipeline

A typical video processing pipeline includes several stages.

### 3.1 Video Sources

Video data can come from:

- Video cameras.
- Synthetic video.
- Pre-recorded video files.

Important properties of video sources:

- Frame rate.
- Resolution.
- File format.
- Compression type.
- Duration.

### 3.2 Capturing and Loading Video Data

Video can be processed in two main ways:

- Reading a live stream frame by frame.
- Loading a stored video file from disk.

Live streams are used when the system must react in real time.

Stored files are used when videos are processed offline, for example during model training.

### 3.3 Pre-processing

Video pre-processing may include:

- Removing sensor noise.
- Removing compression artifacts.
- Correcting camera shake or jitter.
- Applying geometric corrections.
- Adjusting color and lighting.
- Temporal subsampling.

Temporal subsampling means selecting only some frames instead of using every frame.

This is important because raw video is very large and expensive to process.

### 3.4 Feature Representation

Raw video frames are transformed into representations that are useful for the task.

Before deep learning, computer vision systems used handcrafted descriptors.

With deep learning, models learn feature representations automatically.

The goal is to extract patterns that help solve tasks such as:

- Object tracking.
- Activity recognition.
- Scene understanding.
- Video captioning.

---

## 4. The Nature of Video Data

Video can be seen as an image sequence with an additional time dimension.

### 4.1 Tensor Representation

A standard image tensor has the shape:

X_image ∈ R^(C × H × W)

where:

- C = number of channels.
- H = height.
- W = width.

A raw video tensor has the shape:

X_video ∈ R^(C × T × H × W)

where:

- T = number of frames.

A batched video tensor in PyTorch usually has the shape:

X_batch ∈ R^(B × C × T × H × W)

where:

- B = batch size.
- C = channels.
- T = time frames.
- H = height.
- W = width.

Adding the temporal dimension greatly increases the amount of data.

---

## 5. Temporal Redundancy

Video contains a lot of redundant information.

In many videos, most of the background stays almost unchanged between consecutive frames.

For most spatial coordinates (h, w):

X_c,t,h,w ≈ X_c,t-1,h,w

This means that frame-by-frame processing with a 2D CNN repeatedly recomputes similar static background features.

This creates spatial inefficiency.

Example:

If a camera records a static background and only a small object moves, most pixels are repeated from frame to frame.

---

## 6. Hard Drive Bottleneck

Raw video requires a very large amount of storage.

For a 1080p video recorded at 30 FPS:

1920 × 1080 pixels × 3 bytes per pixel × 30 FPS = 186.6 MB/s

If 90% of each frame is redundant static background:

186.6 MB/s × 0.90 = 167.9 MB/s

So the system writes about 167.9 MB of redundant data every second.

For one minute of video from a single camera, this is approximately:

167.9 MB/s × 60 seconds ≈ 10 GB of redundant static data

This makes saving raw video tensors to disk impractical.

For large datasets such as Kinetics-400, raw video storage would require an extremely large amount of storage.

Because of this, industry uses video compression codecs.

This shifts the bottleneck:

- From storage capacity.
- To CPU decoding and frame extraction.

---

## 7. Video Compression with Codecs

Video is usually stored in compressed format.

Compression reduces redundancy by encoding motion and differences between frames.

To use video in deep learning, compressed video must be decoded into:

- Raw frames.
- Motion-related information provided by the codec.

### 7.1 Group of Pictures

Compressed videos are organized into Groups of Pictures, or GOPs.

A GOP contains different types of frames:

- I-frames.
- P-frames.
- B-frames.

### 7.2 I-Frames

I-frame means intra-coded frame.

Properties:

- Stores a full image.
- Does not depend on other frames.
- Acts as an anchor frame.
- Requires more storage than predictive frames.

### 7.3 P-Frames

P-frame means predictive frame.

Properties:

- Stores differences from a previous I-frame or P-frame.
- Depends on earlier frames.
- Requires less storage than I-frames.
- Helps reduce redundancy.

### 7.4 B-Frames

B-frame means bi-predictive frame.

Properties:

- Uses information from both past and future frames.
- Stores interpolated differences.
- Can provide better compression.
- Makes decoding more complex because future frames may be needed.

---

## 8. GOP Structure and Compression Trade-offs

The frame structure affects:

- Coding efficiency.
- Decoding efficiency.
- Error resilience.
- Storage requirements.

### 8.1 Short GOP

A short GOP has more frequent I-frames.

Advantages:

- Better error resilience.
- Easier random access.
- Faster seeking.

Disadvantages:

- Larger file size.
- Less compression efficiency.

### 8.2 More B-Frames

More B-frames usually improve compression because they use both past and future references.

Advantages:

- Smaller file size.
- Better compression efficiency.

Disadvantages:

- More complex decoding.
- Higher latency.
- More dependencies between frames.

### 8.3 More P-Frames

More P-frames reduce storage compared to I-frames.

Advantages:

- Better compression than using many I-frames.
- Simpler than B-frame-heavy structures.

Disadvantages:

- Errors can propagate through dependencies.
- Random access becomes harder than with frequent I-frames.

---

## 9. Motion Vectors

Codecs compress video by estimating motion.

To calculate a P-frame, the encoder:

1. Divides the image into macroblocks.
2. Searches where each block moved compared with a previous anchor frame.
3. Saves a 2D motion vector instead of saving full pixel arrays.

A motion vector has the form:

(dx, dy)

where:

- dx = horizontal displacement.
- dy = vertical displacement.

This is similar to a hardware-level approximation of optical flow.

### 9.1 Engineering Use

Instead of passing large RGB tensors through a 3D convolutional model to learn motion implicitly, some architectures extract pre-computed motion vectors directly from the H.264 stream.

This can:

- Bypass spatial redundancy.
- Reduce computational cost.
- Increase training throughput.

---

## 10. Hardware Limit: VRAM

Video tensors can be too large for GPU memory.

Example setup:

- Batch size B = 4 clips.
- Clip duration = 3 seconds.
- Frame rate = 30 FPS.
- Resolution = 1920 × 1080.
- Channels C = 3.
- Data type = 32-bit float.
- Float32 = 4 bytes per value.

### 10.1 Temporal Volume

3 seconds × 30 FPS = 90 frames per clip

So:

T = 90

### 10.2 Total Number of Elements

B × T × H × W × C

4 × 90 × 1920 × 1080 × 3 = 2,239,488,000 elements

### 10.3 Memory Cost

Each element uses 4 bytes.

2,239,488,000 × 4 = 8,957,952,000 bytes

This is approximately:

8.95 GB

This is only the input tensor.

The actual model would require more VRAM because it also needs memory for:

- Model weights.
- Activations.
- Gradients.
- Optimizer states.

---

## 11. Video Pre-processing: Dimensionality Reduction

To fit video into VRAM, dimensionality must be reduced.

Two main strategies are:

- Temporal subsampling.
- Spatial downsampling.

### 11.1 Temporal Subsampling

Temporal subsampling reduces the number of frames.

Example:

Instead of using every frame, use every second frame.

If the original video has 30 FPS and we use stride k = 2, then only half the frames are selected.

This reduces memory and computation.

General idea:

Sampled frames = Input frames selected every k frames

Advantages:

- Reduces temporal volume.
- Reduces VRAM usage.
- Makes training faster.

Disadvantages:

- May lose important motion information.
- Can miss short actions.

### 11.2 Spatial Downsampling

Spatial downsampling reduces frame resolution.

Example:

A 1080p frame can be resized to 224 × 224.

Advantages:

- Greatly reduces memory cost.
- Makes CNN and transformer models easier to train.
- Preserves enough information for many action recognition tasks.

Disadvantages:

- Fine details may be lost.
- Small objects may become harder to detect.

The goal is to reduce resolution while preserving action-relevant features.

---

## 12. Early CNN Attempts for Video

Initial CNN-based video models often processed frames individually.

The basic idea was:

1. Apply image recognition to each frame.
2. Combine the results across time.

Fusion strategies included:

- Single-frame processing.
- Late fusion.
- Early fusion.
- Slow fusion.

Slow fusion of features across time helped, but improvements over static frame CNNs were modest.

Simply stacking frames as channels does not solve video understanding.

---

## 13. Why 2D Temporal Fusion Fails

Standard 2D convolutions are not designed to understand temporal order.

### 13.1 Late Fusion

In late fusion, the network processes frames separately and combines predictions later.

Problem:

- No real temporal receptive field.
- The model can identify objects but not motion.
- It may know that a door appears, but not that the door is opening.

### 13.2 Early Fusion

In early fusion, several frames are stacked as channels.

Problem:

- Sequential order is weakened or destroyed.
- A 2D kernel sees the stacked channels at the same time.
- It cannot slide through time.
- Time is treated similarly to color channels.

This makes the model weak at learning motion patterns.

---

## 14. 3D Convolutional Operator

To model motion, a 2D convolution kernel can be expanded into a 3D convolution kernel.

A 2D convolution uses a kernel such as:

k × k

A 3D convolution uses a kernel such as:

k × k × k

The third dimension allows the kernel to slide through time.

This lets the model learn features such as:

- Changes over time.
- Motion direction.
- Short-term temporal patterns.
- Action-related dynamics.

---

## 15. Parameter Explosion in 3D CNNs

Expanding from 2D to 3D greatly increases the number of parameters and computations.

For a 3 × 3 2D convolution:

Parameters = 9 × C_in × C_out

For a 3 × 3 × 3 3D convolution:

Parameters = 27 × C_in × C_out

This means:

- 3 times more parameters.
- 3 times more FLOPs.
- More VRAM required.
- Higher risk of overfitting.

The input video tensors are already large, so 3D CNNs are expensive to train.

---

## 16. Overfitting in 4D Video Data

With more parameters, a 3D CNN can easily memorize specific frame sequences.

If the model sees the exact same clip every epoch, it may learn:

- Specific background.
- Specific frame order.
- Specific object position.
- Specific camera viewpoint.

This can produce high training accuracy but poor test accuracy.

The model must learn general action patterns, not memorize a specific 4D sequence.

---

## 17. Fighting Overfitting

The lecture introduces two strategies to avoid showing the exact same clip every time.

### 17.1 Option A: Varying Temporal Stride

The model can be trained with different temporal strides.

Example:

- In one epoch, train on every 2nd frame.
- In another epoch, train on every 4th frame.

This changes the temporal sampling pattern.

Advantages:

- Increases data diversity.
- Reduces memorization.
- Reduces VRAM usage.

### 17.2 Option B: Temporal Jitter

Temporal jitter means randomly changing the start frame of the clip.

Example:

Instead of always starting at frame 0, start at:

- Frame 5.
- Frame 12.
- Frame 18.

This produces different clips from the same video.

Advantages:

- Prevents the model from memorizing fixed sequences.
- Increases effective dataset diversity.
- Helps the model generalize better.

---

## 18. Transformers for Video Understanding

Transformers were introduced into computer vision through Vision Transformers.

A Vision Transformer treats an image as a sequence of patches.

For video, this idea is extended to space-time patches.

---

## 19. Video Transformers

A Video Transformer sees a video clip as a cube of spatiotemporal patches.

### 19.1 Spatiotemporal Patches

Instead of extracting only 2D image patches, the model extracts patches with size:

P × P × T

Each patch contains:

- Spatial texture.
- Temporal change.

These patches are sometimes called tublets.

### 19.2 3D Positional Encoding

The model needs to know where each patch is located.

A learned positional vector represents:

(x, y, t)

where:

- x = horizontal position.
- y = vertical position.
- t = time position or frame index.

This tells the model both:

- Where the patch is inside the frame.
- Which frame it belongs to.

### 19.3 Global Self-Attention

Unlike 3D CNNs, which usually see only a small temporal window, transformers can use global self-attention.

This means that Frame 1 can directly interact with Frame 30.

Advantages:

- Captures long-range temporal dependencies.
- Models relationships between distant frames.
- Can learn complex temporal patterns.

Disadvantages:

- Computationally expensive.
- Requires large datasets or strong pretraining.
- Memory cost can be high.

---

## 20. Video Understanding Tasks

The lecture lists several important video understanding tasks.

### 20.1 Background Subtraction

Background subtraction separates moving foreground objects from a mostly static background.

Basic idea:

1. Estimate a background model.
2. Compare the current frame with the background.
3. Extract a foreground mask.

Useful for:

- Surveillance.
- Motion detection.
- Object segmentation.
- Pre-processing for tracking.

### 20.2 Motion Estimation

Motion estimation calculates how pixels or regions move between frames.

Optical flow is a common example.

Optical flow estimates a motion vector for pixels or regions.

It is useful for:

- Video compression.
- Frame interpolation.
- Object tracking.
- Activity recognition.

### 20.3 Video Object Tracking

Object tracking follows an object across frames.

The goal is to estimate the location of the object over time.

Tracking is useful for:

- Surveillance.
- Autonomous driving.
- Sports analytics.
- Human-computer interaction.
- Retail monitoring.

### 20.4 Human Activity Recognition

Human activity recognition classifies what action is happening in a video.

Examples:

- Walking.
- Running.
- Jumping.
- Opening a door.
- Playing a sport.

This task requires both spatial and temporal understanding.

### 20.5 Video Captioning

Video captioning generates a natural language description of a video.

Example:

Input video:

A person opens a door and enters a room.

Output caption:

“A person opens the door and walks into the room.”

This task combines computer vision and natural language processing.

### 20.6 Temporal Action Localization

Temporal action localization identifies when an action starts and ends in a video.

The output is not only the action class, but also the time interval.

Example:

Action: “jumping”

Start time: 00:05

End time: 00:09

This is more detailed than simple video classification.

---

## 21. Tools for Video Understanding

The lecture mentions several tools and libraries.

### 21.1 OpenCV

OpenCV is commonly used for:

- Reading videos.
- Extracting frames.
- Resizing frames.
- Basic video processing.
- Traditional computer vision algorithms.

### 21.2 Torchvision

Torchvision provides tools for computer vision in PyTorch.

It includes:

- Datasets.
- Transforms.
- Video classification datasets.
- Optical flow datasets.
- Data augmentation utilities.

Useful dataset categories include:

- Optical flow datasets.
- Video classification datasets.

### 21.3 PyTorchVideo

PyTorchVideo is a library designed for video understanding with PyTorch.

It provides support for:

- Video datasets.
- Video models.
- Video transforms.
- Training pipelines.

It can be used with datasets such as:

- Kinetics-400.
- HMDB51.
- UCF101.

---

## 22. Key Takeaways

Video understanding is harder than image understanding because video adds the temporal dimension.

The main challenges are:

- Larger tensor sizes.
- High VRAM usage.
- Temporal redundancy.
- Storage bottlenecks.
- CPU decoding bottlenecks.
- Compression dependencies.
- Risk of overfitting specific clips.

Important solutions include:

- Video compression.
- Temporal subsampling.
- Spatial downsampling.
- Temporal jitter.
- Varying temporal stride.
- 3D convolutions.
- Video transformers.
- Efficient video data loading pipelines.

Video models must learn both:

- What appears in each frame.
- How visual content changes over time.

# Lesson 02: Video Classification

## 1. Objectives

This lecture focuses on video classification: assigning an action label to a video clip.

Main objectives:

- Understand the difference between spatial and temporal feature extraction.
- Analyze the limitations of early 2D CNN approaches for video.
- Understand 3D convolutions and why they are expensive.
- Study I3D, or Inflated 3D ConvNets.
- Understand representation bias in video datasets.
- Explore modern video architectures such as SlowFast, TimeSformer, and ViViT.

---

## 2. What Is Video Classification?

Video classification means predicting the action or event happening in a video clip.

A video is not a single image. It is a sequence of images over time.

A video can be represented as:

T x 3 x H x W

where:

- T = number of frames.
- 3 = RGB channels.
- H = frame height.
- W = frame width.

In image classification, the model usually recognizes objects.

Examples:

- Dog.
- Cat.
- Fish.
- Truck.

In video classification, the model usually recognizes actions.

Examples:

- Swimming.
- Running.
- Jumping.
- Eating.
- Standing.

The important difference is that video classification needs both:

- Spatial information: what appears in the frame.
- Temporal information: how the content changes across frames.

Example:

A single image can show a person.

A video can show whether the person is running, jumping, sitting down, or standing up.

---

## 3. Early Approaches

Early video classification methods tried to reuse image classification CNNs.

The basic idea was:

1. Sample one or more frames from the video.
2. Feed frames into a 2D CNN.
3. Combine the results somehow.
4. Predict one video-level class.

---

## 4. Single-Frame Approach

The simplest approach is to select one frame from the video and classify it with a standard image CNN.

Pipeline:

1. Sample frames from the video.
2. Select a single frame.
3. Feed it to a CNN.
4. Predict a class.

Problem:

A single frame may not contain enough information to understand the action.

Example:

A person standing still and a person about to jump may look similar in one frame.

This approach ignores temporal dynamics.

---

## 5. Multiple-Frame Approach

A better early approach is to process several frames.

The main questions are:

1. What are we aggregating?
2. How do we combine frame-level representations?
3. Where in the network do we combine them?

There are two main types of fusion:

- Score fusion.
- Feature fusion.

---

## 6. Score Fusion

In score fusion, each frame is classified independently.

Each frame produces a score vector.

Then the frame-level score vectors are aggregated into one video-level score vector.

Example:

Frame 1 -> class scores  
Frame 2 -> class scores  
Frame 3 -> class scores  
Final video prediction -> aggregation of frame scores

Advantages:

- Simple.
- Easy to implement.
- Can reuse image classification models.

Disadvantages:

- Temporal order is mostly ignored.
- Motion is not explicitly modeled.
- The model may fail when the same frames in different order mean different actions.

---

## 7. Feature Fusion

In feature fusion, the CNN extracts features from each frame.

Then frame-level features are combined before the final classifier.

Example:

Frame 1 -> CNN features  
Frame 2 -> CNN features  
Frame 3 -> CNN features  
Combined features -> classifier -> video class

Advantages:

- More flexible than score fusion.
- Allows the model to combine visual information before classification.

Disadvantages:

- Still may not model temporal order well.
- Depends heavily on how features are aggregated.

---

## 8. Aggregation Methods

Frame-level features can be aggregated in different ways.

### 8.1 Max Fusion

Max fusion takes the element-wise maximum across frame-level features.

Formula:

y_max = max{x1, x2, ..., xL}

where:

- x1, x2, ..., xL are frame-level feature vectors.
- L is the number of frames.

Meaning:

For each feature dimension, keep the strongest activation across frames.

Advantage:

- Captures whether a visual pattern appeared at least once.

Disadvantage:

- Ignores when the feature appeared.
- Loses temporal order.

---

### 8.2 Average Fusion

Average fusion computes the arithmetic mean of frame-level features.

Formula:

y_avg = (x1 + x2 + ... + xL) / L

Advantage:

- Simple and stable.
- Uses information from all frames.

Disadvantage:

- Smooths out short but important actions.
- Ignores temporal order.

---

### 8.3 Concatenation

Concatenation stacks frame-level features into one long vector.

Formula:

y_concat = {x1; x2; ...; xL}

Advantage:

- Keeps frame features separated.
- Can preserve some order if the input length is fixed.

Disadvantage:

- Works only when every input has the same number of frames.
- Increases feature dimensionality.
- Does not automatically learn temporal motion unless followed by suitable layers.

---

### 8.4 Stacking + 1x1 Convolution

This method stacks frame-level features and applies a 1x1 convolution for dimensionality reduction.

Advantage:

- Reduces the dimensionality of stacked features.
- More learnable than simple average or max fusion.

Disadvantage:

- Still limited if temporal structure is not modeled deeply.

---

## 9. Where to Fuse Information

The lecture presents three levels of fusion:

- Late fusion.
- Early fusion.
- Slow fusion.

---

## 10. Late Fusion

Late fusion processes each frame independently with a 2D CNN.

Then high-level features or predictions are combined near the end.

Pipeline:

Input video:

T x 3 x H x W

Frame-level features:

T x D x H' x W'

Then:

1. Run a 2D CNN on each frame.
2. Concatenate or aggregate frame features.
3. Feed them to an MLP.
4. Produce class scores.

Intuition:

Get high-level appearance features from each frame and combine them.

Problem:

Late fusion is often permutation invariant.

This means the model may not care about frame order.

Example:

Sequence 1:

Person standing -> person sitting down

Sequence 2:

Person sitting -> person standing up

Both sequences may contain similar frames, but the order changes the action.

If the model averages features, it may predict the same class for both.

So late fusion does not properly capture temporal order.

---

## 11. Early Fusion

Early fusion combines frames at the input level.

The video input has shape:

T x 3 x H x W

It is reshaped into:

3T x H x W

Then a standard 2D CNN is applied.

The first 2D convolution receives all frames as channels.

Intuition:

Compare frames with the first convolutional layer, then continue with a normal 2D CNN.

Problem:

The first convolution collapses temporal information immediately.

Input:

3T x H x W

Output after first convolution:

D x H x W

After this first layer, the rest of the network no longer has an explicit time dimension.

This is called temporal squashing.

One layer of temporal processing is usually not enough.

The temporal order can be destroyed because time is treated like extra color channels.

---

## 12. Why Early 2D Approaches Are Limited

Early 2D CNN approaches have several problems:

- Single-frame methods ignore motion completely.
- Score fusion ignores temporal structure.
- Average and max fusion lose frame order.
- Early fusion collapses time too early.
- Late fusion is often insensitive to sequence order.
- 2D convolutions are designed for spatial patterns, not spatio-temporal patterns.

The key issue:

Video is not just a set of images.  
Video is images plus time.

---

## 13. Two-Stream Networks

Two-stream networks were designed to model appearance and motion separately.

They use two branches:

1. Spatial stream.
2. Temporal stream.

---

## 14. Spatial Stream

The spatial stream performs action recognition from still RGB video frames.

Input:

3 x H x W

Purpose:

Capture appearance information.

Examples:

- Person.
- Ball.
- Road.
- Swimming pool.
- Bicycle.

The spatial stream answers:

What is visible in the frame?

---

## 15. Temporal Stream

The temporal stream recognizes action from motion.

It uses dense optical flow as input.

Input:

[2 x (T - 1)] x H x W

Why 2 x (T - 1)?

For each pair of consecutive frames, optical flow has two components:

- dx: horizontal motion.
- dy: vertical motion.

For T frames, there are T - 1 motion transitions.

Purpose:

Capture how pixels move over time.

The temporal stream answers:

How is the scene moving?

---

## 16. Optical Flow

Optical flow gives a displacement field between two consecutive frames.

Between image I_t and image I_t+1, optical flow tells where each pixel moves.

Formula:

F(x, y) = (dx, dy)

where:

- dx = horizontal displacement.
- dy = vertical displacement.

The relationship can be written as:

I_t+1(x + dx, y + dy) = I_t(x, y)

Meaning:

A pixel at location (x, y) in frame t moves to approximately (x + dx, y + dy) in frame t + 1.

Optical flow highlights local motion.

It can be visualized as:

- Horizontal flow dx.
- Vertical flow dy.

---

## 17. Two-Stream Fusion

The spatial stream and temporal stream produce separate class scores.

Then their scores are fused to produce the final action prediction.

Example:

Spatial stream detects:

- Person.
- Gym.
- Equipment.

Temporal stream detects:

- Arm movement.
- Body movement.
- Direction of motion.

Final prediction:

- Jumping.
- Running.
- Long jump.
- Boxing.

---

## 18. Limitations of Two-Stream Networks

Two-stream networks were important, but they have several limitations.

### 18.1 Computational Burden

Optical flow computation is expensive.

Before training or inference, the system must compute optical flow for many frames.

This adds significant processing time.

---

### 18.2 Storage Cost

If optical flow is precomputed and stored, it requires a lot of disk space.

For large video datasets, storing RGB frames plus optical flow is very expensive.

---

### 18.3 No End-to-End Training

Traditional two-stream systems often compute optical flow outside the neural network.

This means the optical flow stage is not trained jointly with the final classification objective.

The model cannot fully optimize the motion representation end to end.

---

### 18.4 Missing Long-Range Temporal Information

Two-stream networks often focus on short-term motion.

They may miss long-range temporal dependencies.

Example:

An action may only be understandable after seeing a longer sequence.

---

### 18.5 False Label Assignment

If the sampled frames miss the actual action moment, the model may receive misleading supervision.

Example:

A video is labeled “Long Jump,” but sampled frames may show only preparation or landing, not the jump itself.

This creates false label assignment.

---

## 19. Native Spatio-Temporal Solution: 3D Convolutions

3D convolutions are a native solution for video because they operate over both space and time.

A 2D convolution slides over:

- Height.
- Width.

A 3D convolution slides over:

- Time.
- Height.
- Width.

This allows the model to learn motion patterns directly from RGB video.

---

## 20. 2D vs 3D Convolution

### 20.1 2D Convolution

For an image, a 2D convolution uses a spatial kernel.

Example:

3 x 3 filter

It moves over the height and width of the image.

For RGB images, the kernel also spans the input channels.

The output is a feature map representing spatial patterns.

---

### 20.2 3D Convolution

For a video, a 3D convolution uses a spatio-temporal kernel.

Example:

3 x 3 x 3 filter

It moves over:

- Time.
- Height.
- Width.

This allows the model to learn features such as:

- Motion direction.
- Temporal changes.
- Short action patterns.
- Object movement.

---

## 21. Output Size Example for 3D Convolution

The lecture gives an example where the final output size is:

3 x 30 x 30 x 4

If padding is 1, the output becomes:

5 x 32 x 32 x 4

Interpretation:

Padding preserves more of the temporal and spatial dimensions.

Without padding, convolution reduces the size because the kernel cannot be centered at the borders.

---

## 22. C3D: The VGG of 3D CNNs

C3D is an early and influential 3D CNN architecture.

It is often described as the VGG-style model for video.

Main characteristics:

- Uses 3 x 3 x 3 convolutions.
- Uses 2 x 2 x 2 pooling.
- Pool1 is an exception.
- Was released with a model pre-trained on Sports-1M.
- Can be used as a video feature extractor.

C3D learns spatio-temporal features directly from video clips.

---

## 23. Problems with C3D

C3D has important limitations.

### 23.1 Overfitting

3D CNNs have many parameters.

Video datasets were historically smaller than image datasets.

This makes 3D CNNs prone to overfitting.

---

### 23.2 Computational Cost

3D convolutions are expensive.

The lecture compares approximate computation:

- AlexNet: 0.7 GFLOP.
- VGG-16: 13.6 GFLOP.
- C3D: 39.5 GFLOP.

C3D is about 2.9 times more expensive than VGG-16.

Reason:

A 3D kernel performs convolution over time, height, and width.

---

## 24. I3D: Inflated 3D ConvNets

I3D means Inflated 3D ConvNet.

The motivation:

There has been a lot of successful work on image architectures.

Question:

Can we reuse image architectures for video?

I3D answers this by converting 2D ConvNets into 3D ConvNets.

---

## 25. Inflating 2D ConvNets into 3D

There are two main ideas.

### 25.1 Option A: Make Square Filters Cubic

A 2D filter has shape:

N x N

It can be inflated into a 3D filter:

N x N x N

Example:

3 x 3 becomes 3 x 3 x 3.

This gives the network temporal receptive fields.

---

### 25.2 Option B: Use 2D Weights to Initialize 3D Weights

A 2D convolution kernel can be copied Kt times along the temporal dimension.

2D kernel:

Cin x Kh x Kw

3D kernel:

Cin x Kt x Kh x Kw

To preserve the activation scale, the copied weights are divided by Kt.

This makes the 3D convolution behave like the original 2D convolution when the video input is constant over time.

---

## 26. Why Divide by Kt in I3D?

Suppose the original 2D case is:

Weight:

W = 6

Input pixel:

X = 2

2D activation:

Y = W x X = 6 x 2 = 12

Now inflate to 3D without scaling:

W_3D = [6, 6, 6]

Static video input:

X_3D = [2, 2, 2]

Activation:

Y_3D = 6 x 2 + 6 x 2 + 6 x 2 = 36

This is too large.

Now apply scaling by 1 / Kt.

If Kt = 3:

W_scaled = [2, 2, 2]

Corrected activation:

Y_correct = 2 x 2 + 2 x 2 + 2 x 2 = 12

Result:

The 3D inflated convolution gives the same output scale as the original 2D convolution.

This preserves the pre-trained distribution.

---

## 27. R(2+1)D: Residual (2+1)D ConvNets

R(2+1)D factorizes a 3D convolution into two separate operations:

1. Spatial 2D convolution.
2. Temporal 1D convolution.

A full 3D convolution has parameters:

Cout x Cin x Kt x Kh x Kw

A (2+1)D block splits it into:

Spatial convolution:

Cout x M x 1 x Kh x Kw

Temporal convolution:

M x Cin x Kt x 1 x 1

where:

- M is the number of intermediate channels.
- M is a hyperparameter.

---

## 28. Why Use (2+1)D Convolutions?

A full 3D convolution mixes space and time in one operation.

A (2+1)D convolution separates the learning process:

1. Learn spatial patterns.
2. Learn temporal patterns.

Advantages:

- Fewer parameters.
- Lower computation.
- Extra non-linearity can be inserted between spatial and temporal convolutions.
- Often easier to optimize.

---

## 29. Parameter Reduction Example

Given:

- Input channels C = 64.
- Output channels C = 64.
- Spatial kernel = 3 x 3.
- Temporal kernel Kt = 3.

Full 3D convolution:

64 x 64 x 3 x 3 x 3 = 110,592 parameters

(2+1)D convolution:

Spatial part:

64 x 64 x 1 x 3 x 3 = 36,864

Temporal part:

64 x 64 x 3 x 1 x 1 = 12,288

Total:

36,864 + 12,288 = 49,152 parameters

Reduction:

Approximately 55% fewer parameters for a single layer.

---

## 30. Cognitive Reset: Real-Time Fall Detection Example

Scenario:

A C3D model is deployed for real-time fall detection in a hospital corridor.

Input:

- 16-frame clips.
- 30 FPS.
- 224 x 224 resolution.
- Float32.
- Current latency: 340 ms per clip.
- Required latency: under 100 ms.

Proposed fixes:

A. Reduce temporal stride from 1 to 4, keeping resolution 224 x 224.  
B. Reduce spatial resolution from 224 x 224 to 112 x 112, keeping stride 1.  
C. Replace C3D with a (2+1)D factorized network at the same channel width.

Expected latency reduction:

1. B: Reducing spatial resolution from 224 x 224 to 112 x 112 gives a large reduction because spatial area becomes 4 times smaller.
2. A: Increasing temporal stride reduces the number of frames processed, so it also reduces computation significantly.
3. C: Replacing C3D with (2+1)D reduces parameters and computation, but the effect depends on implementation and architecture.

Highest accuracy risk for fall detection:

A has the highest risk.

Reason:

Fall detection depends strongly on temporal dynamics.

If temporal stride is too large, the model may skip critical frames showing the fall transition.

A fall can happen quickly, so reducing temporal sampling may remove the most important evidence.

---

## 31. SlowFast Networks

SlowFast is a modern architecture designed to capture both semantic appearance and fast motion.

It has two pathways:

- Slow pathway.
- Fast pathway.

---

## 32. Slow Pathway

The Slow pathway operates at a low frame rate.

Purpose:

Capture spatial semantics.

It focuses on appearance and high-level context.

Characteristics:

- Uses a large temporal stride tau.
- Processes only one out of tau frames.
- Typical value: tau = 16.
- For 30 FPS video, this is roughly 2 frames per second.

The Slow pathway answers:

What is happening at a semantic level?

---

## 33. Fast Pathway

The Fast pathway operates at a high frame rate.

Purpose:

Capture motion at fine temporal resolution.

Characteristics:

- Uses a smaller temporal stride tau / alpha.
- alpha > 1 is the frame-rate ratio between the Fast and Slow pathways.
- Has high temporal resolution.
- Avoids temporal downsampling layers.
- Uses no temporal pooling or time-strided convolutions.
- Has lower channel capacity.

The Fast pathway answers:

How is motion changing over time?

---

## 34. SlowFast Channel Ratio

The Fast pathway uses fewer channels than the Slow pathway.

If the Slow pathway has C channels, the Fast pathway has:

beta x C

where:

- beta < 1.
- Typical beta = 1/8.

This makes the Fast pathway lightweight.

Even though it processes more frames, it has low channel capacity.

The lecture notes that the Fast pathway accounts for nearly 20% of the total computation.

---

## 35. Why SlowFast Works

The two pathways specialize in different information.

Slow pathway:

- High channel capacity.
- Low temporal resolution.
- Strong spatial semantics.

Fast pathway:

- Low channel capacity.
- High temporal resolution.
- Strong motion representation.

Example from the lecture:

Slow pathway:

- 8 frames.
- High channel capacity, such as 64 channels.

Fast pathway:

- 32 frames.
- Low channel capacity, such as 8 channels.

The Fast pathway has many more temporal steps, so temporal convolution is meaningful throughout the network.

---

## 36. Lateral Connections in SlowFast

SlowFast fuses information from the Fast pathway into the Slow pathway.

This is done using lateral connections.

The lateral connection must solve two problems:

1. Reduce T_fast to match T_slow.
2. Project channels to match the expected input of the next Slow block.

Example:

Fast pathway:

T_fast = 32

Slow pathway:

T_slow = 8

The lateral connection reduces temporal resolution with stride 4:

32 -> 8

Then it projects channels so that the fused representation can be used by the Slow pathway.

---

## 37. Transformers for Video

Transformers use self-attention to model relationships between tokens.

For images, Vision Transformers split an image into patches.

For videos, transformers must handle both:

- Spatial patches.
- Temporal frames.

This creates a much larger number of tokens.

---

## 38. Attention Cost for Images vs Videos

For a standard image:

- Resolution: 224 x 224.
- Patch size: 16 x 16.

Number of patches:

14 x 14 = 196 tokens

Attention matrix size:

196^2 = 38,416

For a video:

- 16 frames.
- Resolution: 224 x 224.
- Patch size: 16 x 16.

Tokens per video:

16 x 196 = 3,136 tokens

Attention matrix size:

3,136^2 ≈ 10 million

Problem:

Global spatio-temporal attention is very expensive for video.

---

## 39. TimeSformer: Factorized Attention

TimeSformer reduces the cost of video attention by factorizing it.

Instead of applying full global spatio-temporal attention, it separates:

- Spatial attention.
- Temporal attention.

Given:

- T = 16 frames.
- P = 196 spatial patches.

Global spatio-temporal attention cost:

(T x P)^2

(16 x 196)^2 = 3,136^2 ≈ 10 million operations

Divided space-time attention cost:

T x P^2 + P x T^2

Spatial attention:

16 x 196^2 ≈ 615K

Temporal attention:

196 x 16^2 ≈ 50K

Total:

≈ 665K

Result:

Approximately 93.2% reduction in memory footprint.

---

## 40. Temporal Attention in TimeSformer

For temporal attention:

1. Merge batch and space dimensions.
2. The attention module sees B x S independent sequences.
3. Each sequence has length T.
4. Run standard self-attention over time.
5. Reshape back to the original dimensions.

Meaning:

For each spatial location, the model attends across frames.

This helps the model understand how the same patch changes over time.

---

## 41. Spatial Attention in TimeSformer

For spatial attention:

1. Merge batch and time dimensions.
2. The attention module sees B x T independent sequences.
3. Each sequence has length S.
4. Run standard self-attention over spatial patches.

Meaning:

For each frame, the model attends across image patches.

This helps the model understand spatial relationships inside each frame.

---

## 42. ViViT and Tubelet Embeddings

ViViT is a video transformer architecture.

Instead of treating every frame patch independently, ViViT can use tubelet embeddings.

A tubelet is a spatio-temporal patch.

It covers:

- A spatial region.
- Several consecutive frames.

This reduces the number of tokens and gives each token temporal information from the start.

---

## 43. ViViT Tokenization

The lecture gives an example output shape before flattening:

[2, 768, 8, 14, 14]

This is a 5D video volume.

A transformer cannot directly accept this 5D tensor.

A transformer expects a 3D sequence tensor:

[Batch, Sequence_Length, Embedding_Dimension]

So the spatial and temporal dimensions must be flattened.

Number of tokens:

N = Time x Height x Width

N = 8 x 14 x 14 = 1,568 tokens

Final tensor shape:

[2, 1568, 768]

where:

- 2 = batch size.
- 1568 = sequence length.
- 768 = embedding dimension.

---

## 44. Benchmarking on Kinetics-400

The lecture shows top-1 accuracy on Kinetics-400 for several model families.

Approximate values:

- Per-frame CNN: 62.2
- CNN + LSTM: 63.3
- Two-Stream CNN: 65.6
- I3D: 71.1
- Inflated I3D: 74.2
- SlowFast 16x8 + NL: 79.8
- MViTv2-B, 32x3: 82.9
- MViTv2-L: 86.1
- Very large modern model variant: around 90

Main trend:

Models that better capture spatio-temporal structure perform better.

Early 2D CNN methods are weaker.

Modern architectures such as SlowFast and video transformers achieve stronger performance.

---

## 45. Datasets

The lecture introduces several important video classification datasets.

---

## 46. UCF101

UCF101 is a video action recognition dataset.

Properties:

- YouTube videos.
- 101 action classes.
- 13,320 videos.
- Around 27 hours of video data.
- Large variation in camera motion.
- Large variation in object appearance and pose.
- Large variation in viewpoint, background, and illumination.

An older baseline using HOG/HOF descriptors with SVM achieved about 43.9% accuracy.

---

## 47. Sports-1M

Sports-1M is a large-scale sports video dataset.

Properties:

- Around 1 million YouTube videos.
- 487 classes.
- 1000 to 3000 videos per class.
- Around 5% of videos have more than one class label.

Example classes include:

- Boomerang.
- Boxing.
- Bowling.
- Cycling.
- Skittles.
- Ten-pin bowling.

Sports-1M was used to pre-train C3D.

---

## 48. YouTube-8M

YouTube-8M is a large-scale video classification benchmark.

Properties:

- Millions of videos.
- Thousands of classes.
- Multi-label classification.
- Around 500,000 hours of video.
- Vocabulary of around 4,800 visual entities.

It is designed for general video understanding at large scale.

---

## 49. Kinetics

Kinetics is a large human action recognition dataset.

Properties:

- YouTube videos.
- 650,000 video clips.
- Covers 400, 600, or 700 human action classes depending on the version.
- At least 600 video clips per action class.
- Each clip lasts around 10 seconds.
- Each clip is labeled with a single action class.

Kinetics is widely used for benchmarking video classification models.

---

## 50. From Video Classification to Temporal Action Localization

So far, video classification assumes short trimmed clips.

In trimmed video classification:

Input:

Short clip

Output:

One action label

Example:

Video clip -> Running

But real videos are often long and untrimmed.

In long videos, the task becomes more complex.

---

## 51. Temporal Action Localization

Temporal action localization means identifying where actions happen in time.

Given a long untrimmed video, the model must identify frames or time intervals corresponding to different actions.

Example:

From a long video:

- Running from 00:05 to 00:10.
- Jumping from 00:11 to 00:13.

The model predicts:

- Action class.
- Start time.
- End time.

This is similar in spirit to object detection, but along the temporal dimension.

A possible approach:

1. Generate temporal proposals.
2. Classify each proposal.

This is similar to Faster R-CNN, but applied to time intervals instead of image regions.

---

## 52. Spatio-Temporal Detection

Spatio-temporal detection is even more detailed.

Given a long untrimmed video, the model must:

1. Detect people in space.
2. Track them through time.
3. Classify the activities they are performing.

Output:

- Bounding boxes.
- Time intervals.
- Action labels.

Example:

A person is detected across multiple frames and classified as:

- Walking.
- Sitting.
- Talking.
- Running.

Datasets such as AVA are used for this type of task.

---

## 53. Key Comparison of Architectures

### Single-Frame CNN

Uses one frame.

Strength:

- Simple.
- Cheap.

Weakness:

- Ignores motion.

---

### Late Fusion

Processes frames separately and combines results near the end.

Strength:

- Reuses image CNNs.
- Simple to implement.

Weakness:

- Often ignores temporal order.

---

### Early Fusion

Stacks frames as channels.

Strength:

- Allows the first layer to compare frames.

Weakness:

- Temporal information is collapsed too early.

---

### Two-Stream Network

Uses RGB frames and optical flow.

Strength:

- Explicitly models appearance and motion.

Weakness:

- Expensive optical flow computation.
- High storage cost.
- Not fully end-to-end.

---

### C3D

Uses 3D convolutions directly on video.

Strength:

- Learns spatio-temporal features.

Weakness:

- Very expensive.
- Prone to overfitting.

---

### I3D

Inflates 2D image models into 3D video models.

Strength:

- Reuses image pretraining.
- Preserves useful 2D representations.

Weakness:

- Still computationally heavy.

---

### R(2+1)D

Factorizes 3D convolution into spatial and temporal parts.

Strength:

- Fewer parameters.
- Easier optimization.

Weakness:

- More complex architecture design.

---

### SlowFast

Uses separate slow and fast pathways.

Strength:

- Captures both semantics and motion.
- Efficient because the fast pathway has low channel capacity.

Weakness:

- Requires careful temporal alignment and lateral fusion.

---

### TimeSformer

Uses factorized space-time attention.

Strength:

- Reduces attention cost.
- Captures long-range dependencies.

Weakness:

- Still requires many tokens and strong computation.

---

### ViViT

Uses video transformer tokenization, including tubelets.

Strength:

- Converts video into transformer-compatible token sequences.
- Can model spatio-temporal relationships.

Weakness:

- Memory cost can still be high.
- Requires large-scale data or pretraining.

---

## 54. Key Takeaways

Video classification is harder than image classification because the model must understand both appearance and motion.

Main challenges:

- Video has an additional temporal dimension.
- Frame order matters.
- Actions may depend on short or long temporal patterns.
- 2D CNNs are not naturally designed for time.
- Optical flow is useful but expensive.
- 3D convolutions are powerful but computationally heavy.
- Transformers can model long-range dependencies but attention cost grows quickly.

Main architectural progression:

1. Single-frame CNNs.
2. Multi-frame fusion.
3. Two-stream networks.
4. 3D CNNs.
5. I3D.
6. R(2+1)D.
7. SlowFast.
8. TimeSformer.
9. ViViT.

The central goal of video classification is to build models that can efficiently learn:

- What appears in the video.
- How it moves.
- When the important action happens.

# Lesson 03: Motion Estimation and Object Tracking

## 1. Objectives

This lecture covers two connected topics:

1. Motion estimation.
2. Object tracking.

Main objectives:

- Understand the principles of motion estimation.
- Understand the role of motion estimation in object tracking.
- Learn classical and deep-learning-based motion estimation techniques.
- Explore traditional, learning-based, and deep-learning-based object tracking methods.
- Compare advantages and limitations of different tracking approaches.
- Understand how these methods are evaluated on real-world videos.

---

# Part I: Motion Estimation

## 2. What Is Motion Estimation?

Motion estimation means estimating how pixels, regions, or objects move between frames in a video.

In video, motion is often one of the most important cues.

Sometimes motion is the only cue that allows us to perceive an object or understand what is happening.

Examples:

- A moving object can become visible even if its appearance is weak.
- Sparse moving points can create a strong perception of a human body.
- Tracking a person or vehicle requires estimating how it moves from frame to frame.

---

## 3. Why Motion Estimation Matters

Motion estimation is important in several areas.

### 3.1 Video Understanding

Motion estimation helps models understand movement in a scene.

It is essential for:

- Video understanding.
- Object tracking.
- Human motion analysis.
- Autonomous navigation.
- Video surveillance.

Motion tells the system how the scene changes over time.

---

### 3.2 Video Compression

Motion estimation is used in video compression.

The goal is to reduce temporal redundancy.

Instead of storing every frame independently, compression algorithms estimate how parts of the image move between frames.

This is called motion-compensated prediction.

It helps reduce:

- Storage requirements.
- Streaming bandwidth.
- Redundant pixel information.

Example:

If a block moves from one position to another, the video codec can store the motion vector instead of storing the full block again.

---

### 3.3 Video Processing

Motion estimation is also used to improve video quality.

Applications include:

- Frame rate conversion.
- De-interlacing.
- Video stabilization.
- Reducing camera shake.
- Improving clarity and realism.

---

## 4. Classical Motion Estimation

The classical approach to motion estimation is based on optical flow.

Optical flow estimates the 2D motion of every pixel between consecutive frames.

Output:

For each pixel, optical flow predicts a motion vector:

F(x, y) = (dx, dy)

where:

- dx = horizontal displacement.
- dy = vertical displacement.

---

## 5. Optical Flow

Optical flow describes how image points move between frames.

Given two frames:

- I_t: image at time t.
- I_t+1: image at time t + 1.

Optical flow estimates where each pixel from I_t moves in I_t+1.

A pixel at location:

(x, y)

moves to:

(x + dx, y + dy)

The optical flow vector is:

(dx, dy)

---

## 6. Brightness Constancy Assumption

The fundamental assumption of optical flow is brightness constancy.

It means that the appearance of a moving point stays approximately the same between frames.

Mathematically:

I_t(p) ≈ I_t+1(p + w_p)

where:

- p is a pixel location.
- w_p is the motion vector for that pixel.

In coordinate form:

I(x, y, t) = I(x + delta_x, y + delta_y, t + delta_t)

Meaning:

The same physical point has the same brightness after it moves.

---

## 7. Matching-Based Motion Estimation

A simple way to estimate motion is to search for the most similar pixel or patch in the next frame.

For a pixel p in image 1, find a displacement w_p that minimizes the difference:

min over w_p of (I_t(p) - I_t+1(p + w_p))^2

This compares the pixel in the first image with possible locations in the second image.

The displacement with the smallest difference is selected as the motion estimate.

---

## 8. Taylor Approximation and Optical Flow Constraint

The exact brightness constancy equation is nonlinear:

I(x, y, t) = I(x + delta_x, y + delta_y, t + delta_t)

To solve for velocity, it is linearized using a first-order Taylor expansion:

I(x + delta_x, y + delta_y, t + delta_t)
≈ I(x, y, t) + I_x delta_x + I_y delta_y + I_t delta_t

Subtracting I(x, y, t) and dividing by delta_t gives the optical flow constraint equation:

I_x u + I_y v + I_t = 0

where:

- I_x = image gradient in x direction.
- I_y = image gradient in y direction.
- I_t = temporal gradient.
- u = delta_x / delta_t.
- v = delta_y / delta_t.

The velocity vector is:

(u, v)

---

## 9. Aperture Problem

The optical flow constraint equation gives one equation with two unknowns:

I_x u + I_y v = -I_t

Unknowns:

- u
- v

This means there are infinitely many possible velocity vectors that satisfy the equation.

This is called the aperture problem.

### 9.1 Observable Motion

Only motion perpendicular to an edge can be observed clearly.

This is called normal flow.

### 9.2 Invisible Motion

Motion parallel to an edge may produce no visible brightness change.

If motion is parallel to the edge:

I_x u + I_y v = 0

and:

I_t = 0

So the motion becomes invisible to the local optical flow constraint.

---

## 10. Solving Ambiguity with Lucas-Kanade

Lucas-Kanade solves the ambiguity by assuming that pixels in a local patch share the same motion.

Instead of estimating motion for one pixel independently, it estimates one motion vector for a small neighborhood.

The objective becomes:

min over w_p of sum over q in N_p of (I_t(q) - I_t+1(q + w_p))^2

where:

- N_p is the patch around pixel p.
- q is a pixel inside the patch.
- w_p is the shared motion vector.

Assumption:

All pixels in the patch move together.

This gives more equations and makes the motion estimate more stable.

---

## 11. Patch Matching and Cost Volume

Patch matching compares a patch in the first image with candidate patches in the second image.

The result can be represented as a cost volume.

Cost volume:

- Stores matching costs for possible displacements.
- Darker regions usually mean more similar patches.
- Lower cost means better match.

Cost volumes are important in both classical and deep-learning-based optical flow methods.

---

## 12. Effect of Patch Size

Patch size affects optical flow estimation.

### 12.1 Small Patches

Advantages:

- Better at capturing fine details.
- Better near motion boundaries.

Disadvantages:

- More ambiguous.
- More sensitive to noise.
- May fail in textureless regions.

### 12.2 Large Patches

Advantages:

- More stable.
- Less sensitive to noise.
- More context for matching.

Disadvantages:

- Can blur motion boundaries.
- May incorrectly assume that different objects move together.

The patch size creates a trade-off between precision and stability.

---

## 13. Computational Problem: Brute Force Is Expensive

Brute-force matching is expensive because each patch in image 1 may need to be compared with many possible patches in image 2.

For high-resolution images, this becomes very slow.

The search space grows quickly with:

- Image size.
- Patch size.
- Search radius.
- Number of displacement candidates.

---

## 14. Coarse-to-Fine Iterative Estimation

To reduce computation and handle large motion, classical methods use image pyramids.

The idea:

1. Create lower-resolution versions of both images.
2. Estimate motion at the coarsest level.
3. Upsample the estimate to the next finer level.
4. Refine the motion estimate.
5. Repeat until full resolution.

This is called coarse-to-fine estimation.

### 14.1 Why Start Coarse?

Large motion becomes smaller at low resolution.

Example:

A 5-pixel displacement at full resolution may become:

- 2.5 pixels at half resolution.
- 1.25 pixels at quarter resolution.

This makes matching easier.

---

## 15. Warping

Warping uses the current motion estimate to align one image with another.

If we estimate that pixels move by w, we can warp image 2 toward image 1.

Then the remaining difference is smaller.

The algorithm can estimate residual motion iteratively.

Basic idea:

1. Estimate approximate flow.
2. Warp the second image using the current flow.
3. Estimate correction flow.
4. Add correction to the previous flow.
5. Repeat.

---

## 16. Motion Boundaries

A challenge for patch-based methods is motion boundaries.

Motion boundaries occur where two neighboring regions move differently.

Example:

- A moving car in front of a static background.
- A walking person in front of a wall.

Problem:

Lucas-Kanade assumes pixels in a patch share the same motion.

At motion boundaries, this assumption is violated.

A patch may contain pixels from both foreground and background, leading to incorrect flow.

---

## 17. Horn-Schunck Optical Flow

Horn-Schunck solves optical flow by adding a smoothness assumption.

Assumption:

Neighboring pixels usually have similar motion.

This leads to an energy minimization problem.

The method balances:

1. Data term: brightness constancy.
2. Smoothness term: neighboring flow vectors should be similar.

General idea:

Minimize:

brightness constancy error + smoothness penalty

Strength:

- Produces dense optical flow.
- Encourages globally smooth motion fields.

Weakness:

- Can oversmooth motion boundaries.
- Struggles with large motion, occlusion, and illumination changes.

---

## 18. Challenges for Classical Methods

Classical optical flow methods struggle with:

- Large motion.
- Motion blur.
- Occlusions.
- Light changes.
- Noise.
- Textureless areas.
- Repetitive patterns.
- Motion boundaries.

It is difficult to design an objective function that handles all cases.

It is even harder to optimize such objective functions efficiently.

---

# Part II: Deep-Learning-Based Motion Estimation

## 19. Supervised Optical Flow

Deep-learning-based optical flow methods learn to predict flow from data.

Input:

- Two consecutive images.

Output:

- Dense optical flow field.

Training requires ground-truth optical flow.

---

## 20. FlowNet

FlowNet was one of the first deep-learning approaches for optical flow.

It was introduced by Dosovitskiy et al. in 2015.

FlowNet learns a mapping from image pairs to optical flow.

There are two main variants:

- FlowNetS.
- FlowNetC.

---

## 21. FlowNetS

FlowNetS means FlowNet Simple.

It takes two images as input and directly predicts optical flow.

Architecture:

- Similar to U-Net.
- Encoder extracts features.
- Decoder upsamples and predicts dense flow.

Input:

Image 1 + Image 2

Output:

Optical flow.

Strength:

- Simple end-to-end architecture.
- Learns optical flow directly from data.

Weakness:

- Initially behind classical state-of-the-art methods.

---

## 22. FlowNetC

FlowNetC means FlowNet Correlation.

It explicitly compares features from the two images.

Key idea:

- Extract features from both images.
- Compute correlations between features.
- Use these correlations to estimate flow.

This resembles classical cost-volume matching, but with learned features.

---

## 23. FlowNet2

FlowNet2 improves FlowNet by stacking multiple FlowNetS and FlowNetC networks.

It scales up the architecture.

Main idea:

- Use multiple stages.
- Refine flow predictions progressively.
- Combine different FlowNet variants.

Result:

- Significant improvement over the original FlowNet.
- Better accuracy.
- Higher computational cost.

---

## 24. Accuracy and Speed Trade-off

Optical flow models must balance:

- Accuracy.
- Runtime.
- Model size.
- Memory consumption.

Examples of optical flow methods mentioned:

- FlowNet2.
- S2F-IF.
- FlowFieldsCNN.
- MRFlow.
- DCFlow.
- SpyNet.
- PWC-Net.

Different models choose different points on the accuracy-speed trade-off.

---

## 25. PWC-Net

PWC-Net is inspired by classical optical flow methods.

PWC stands for:

- Pyramid.
- Warping.
- Cost volume.

It does not refer to PricewaterhouseCoopers.

Main components:

1. Pyramid of learnable features.
2. Warping using current flow estimates.
3. Cost volume computed by correlation.
4. Mapping cost volume to optical flow.

PWC-Net combines classical ideas with deep learning.

It is more compact and efficient than very large FlowNet-style models.

---

## 26. Learnable Feature Pyramid

Instead of using raw image pyramids, PWC-Net builds pyramids of learned features.

At each level:

- Features are extracted at a certain resolution.
- Flow is estimated from coarse to fine.
- Higher-resolution levels refine the flow.

This helps handle large displacement efficiently.

---

## 27. Cost Volume by Correlation

PWC-Net computes a cost volume by correlating features from the two frames.

For each location, it compares the feature vector with nearby feature vectors in the second image.

This produces matching scores.

The cost volume tells the network where the best match may be.

---

## 28. Iterative Residual Refinement

Iterative Residual Refinement improves optical flow estimation by repeatedly refining the result.

Instead of predicting the final flow in one step, the model predicts corrections.

Process:

1. Start with an initial flow estimate.
2. Warp features or image using current flow.
3. Estimate residual correction.
4. Add correction to flow.
5. Repeat.

This resembles classical iterative optimization.

---

## 29. RAFT

RAFT means Recurrent All-Pairs Field Transforms.

It was introduced by Teed and Deng in 2020.

RAFT is a strong deep-learning method for optical flow.

Main ideas:

- Compute all-pairs visual similarity.
- Build a cost volume.
- Use a cost volume pyramid.
- Recurrently update the flow estimate.

---

## 30. RAFT All-Pairs Similarity

RAFT computes inner products or correlations between features at all pairs of pixels.

This creates a 4D cost volume.

For each pixel in image 1, the model stores similarity to every pixel in image 2.

This allows the model to consider many possible matches.

---

## 31. RAFT Cost Volume Pyramid

RAFT builds a pyramid from the all-pairs cost volume using spatial pooling.

The cost volume pyramid allows the model to access matching information at different scales.

This helps with:

- Large motion.
- Fine motion refinement.
- Robust matching.

---

## 32. RAFT Recurrent Update

RAFT updates the optical flow estimate recurrently.

At each iteration:

1. Use the current motion estimate.
2. Look up relevant values in the cost volume.
3. Update the flow using a recurrent unit.
4. Repeat.

This is similar to classical optimization, but the update rule is learned.

---

## 33. RAFT Hardware Bottleneck

RAFT can be memory-intensive because of the all-pairs cost volume.

For 1080p video:

Resolution:

1920 x 1080 = 2,073,600 pixels per frame

Cost volume elements:

(H x W) x (H x W)

2,073,600 x 2,073,600 ≈ 4.3 trillion elements

If stored as FP32:

4.3 x 10^12 x 4 bytes ≈ 17.2 TB

This is impossible to store directly on normal GPUs.

Therefore, practical implementations require lower resolution, optimized memory usage, or architectural tricks.

---

## 34. Recent Developments in Optical Flow

Recent methods use attention and transformer-based architectures.

Examples:

- Perceiver IO.
- FlowFormer.
- GM-Flow.

These methods try to improve motion estimation by using global matching, attention, and better structured representations.

---

## 35. Training Data for Optical Flow

Architecture alone is not enough.

Optical flow performance also depends strongly on training data.

Common benchmarks:

### 35.1 Sintel

Sintel is based on a Blender movie.

It contains challenging synthetic scenes with:

- Motion blur.
- Atmospheric effects.
- Complex motion.
- Occlusions.

### 35.2 KITTI

KITTI is a driving dataset.

It contains real-world road scenes.

It is important for:

- Autonomous driving.
- Vehicle motion.
- Outdoor optical flow.

---

# Part III: Object Tracking

## 36. What Is Object Tracking?

Object tracking means identifying an object in one frame and estimating where it moves or how it appears in the next frames.

Input:

- Initial object location, often a bounding box.

Output:

- Object location in each subsequent frame.

Tracking can involve:

- Stationary objects.
- Moving objects.
- Changing appearance.
- Occlusion.
- Scale changes.
- Rotation.
- Deformation.

---

## 37. Why Not Just Use Object Detection?

Object detection finds objects independently in each frame.

Object tracking uses temporal continuity.

Tracking is useful because:

- It can be faster than running detection on every frame.
- It preserves object identity across time.
- It can handle temporary missed detections.
- It uses motion and appearance history.
- It can follow a specific target object.

Detection answers:

Where are the objects in this frame?

Tracking answers:

Where did this specific object go over time?

---

## 38. Object Tracking Applications

Object tracking is used in:

- Surveillance.
- Autonomous driving.
- Robotics.
- Sports analytics.
- Human-computer interaction.
- Medical video analysis.
- Retail analytics.
- Traffic monitoring.
- Animal tracking.

---

## 39. Object Tracking Challenges

Object tracking is difficult because of:

- Occlusion.
- Motion blur.
- Illumination changes.
- Scale changes.
- Rotation.
- Deformation.
- Background clutter.
- Fast motion.
- Similar-looking objects.
- Camera motion.
- Out-of-view movement.
- Re-identification after disappearance.

---

# Part IV: Traditional Tracking Methods

## 40. Categories of Traditional Trackers

Traditional tracking methods include:

### 40.1 Feature-Based Tracking

Examples:

- SIFT.
- SURF.
- ORB.

These methods track distinctive visual features.

### 40.2 Model-Based Tracking

Examples:

- Geometric-based tracking.
- Kalman filter.

These methods use a motion model to predict where the object will be.

### 40.3 Appearance-Based Tracking

Examples:

- Mean Shift.
- CamShift.
- Template Matching.

These methods track an object based on how it looks.

---

## 41. Kalman Filter

The Kalman filter is a recursive estimator.

It predicts the state of an object and updates the prediction using observations.

State may include:

- Position.
- Velocity.
- Acceleration.

Basic steps:

1. Prediction step.
2. Measurement update step.

### 41.1 Prediction Step

The motion model predicts where the object should be in the next frame.

Example:

If the object is moving with constant velocity, its next position can be predicted from its previous position and velocity.

### 41.2 Update Step

When a measurement is available, the filter corrects the prediction.

The final estimate combines:

- Predicted state.
- Observed measurement.
- Uncertainty of both.

Strengths:

- Fast.
- Lightweight.
- Useful when motion is smooth.
- Handles noisy measurements.

Weaknesses:

- Assumes a simple motion model.
- Struggles with abrupt motion.
- Does not directly understand image appearance.

---

## 42. Particle Filter

A particle filter represents uncertainty using many samples called particles.

Each particle is a possible object state.

Process:

1. Generate particles around the previous estimate.
2. Predict their next states.
3. Assign weights according to how well each particle matches the observation.
4. Resample particles.
5. Estimate the object state from weighted particles.

Strengths:

- Can model non-linear motion.
- Can represent multi-modal uncertainty.

Weaknesses:

- More computationally expensive than Kalman filtering.
- Requires enough particles for good accuracy.

---

## 43. Mean Shift

Mean Shift is a clustering algorithm.

It is also called a mode-seeking algorithm.

It is non-parametric.

Goal:

Find the region with the highest density.

In tracking, Mean Shift searches for the region most similar to the target appearance, often using a color histogram.

Basic idea:

1. Represent the target using an appearance distribution.
2. Search nearby regions in the next frame.
3. Move the window toward the region with the highest similarity.
4. Repeat until convergence.

Strengths:

- Simple.
- Efficient.
- Works when color distribution is distinctive.

Weaknesses:

- Fixed window size.
- Struggles with scale change.
- Sensitive to background with similar colors.

---

## 44. CamShift

CamShift means Continuously Adaptive Mean Shift.

It extends Mean Shift.

Main improvement:

- The tracking window can adapt its size.

This helps when the object scale changes.

Strength:

- Better than Mean Shift for objects moving toward or away from the camera.

Weakness:

- Still sensitive to appearance ambiguity.
- Can fail under occlusion or background clutter.

---

## 45. Optical Flow-Based Tracking: KLT Tracker

KLT stands for Kanade-Lucas-Tomasi tracker.

It tracks feature points using optical flow.

It combines:

- Lucas-Kanade motion estimation.
- Tomasi-Kanade feature selection.

Assumptions:

- Motion is small.
- Motion is smooth.
- Brightness stays constant.

Strengths:

- Fast.
- Lightweight.
- Good for tracking points.

Weaknesses:

- Tracks sparse features, not full objects.
- Can fail with large motion, occlusion, or low texture.

---

## 46. KLT: How Features Are Tracked

Lucas-Kanade aligns patches between consecutive frames.

For each selected feature point, it estimates a displacement vector.

This tells where the feature moved from one frame to the next.

---

## 47. KLT: Good Features to Track

Tomasi and Kanade introduced a criterion for choosing good features.

Good features are locations where image gradients vary in multiple directions.

Examples:

- Corners.
- Textured regions.
- Distinct local structures.

Bad features:

- Flat regions.
- Single straight edges.
- Textureless areas.

Reason:

Corners provide enough information to estimate motion in both x and y directions.

---

# Part V: Advanced Tracking Methods

## 48. Shift Toward Learned Appearance Models

Advanced trackers move from motion-only models to learned appearance models.

The tracker adapts online to changes in the target.

This helps with:

- Object deformation.
- Appearance changes.
- Drift.
- Partial occlusion.
- Illumination changes.

Tracking can be formulated as:

Tracking = Motion model + Appearance model

---

## 49. Motion Model vs Appearance Model

### 49.1 Motion Model

The motion model predicts likely object location.

Examples:

- Kalman filter.
- Constant velocity model.

It does not depend directly on image content.

It answers:

Where is the object likely to be?

### 49.2 Appearance Model

The appearance model verifies whether the predicted region looks like the target.

Examples:

- Histogram matching.
- Online classifier.
- Correlation filter.

It learns what the object looks like over time.

It answers:

Does this region look like the object?

---

## 50. Learning-Based Trackers

Learning-based trackers update appearance models online.

They adapt to:

- Pose changes.
- Lighting changes.
- Partial occlusion.
- Deformation.

Examples:

- MIL tracker.
- BOOSTING tracker.
- TLD tracker.

---

## 51. MIL Tracker

MIL means Multiple Instance Learning.

The MIL tracker learns an online classifier from bags of instances.

A positive bag contains several patches near the predicted object location.

Only one patch in the positive bag may be the true object.

The tracker learns to resolve this ambiguity over time.

Strengths:

- Handles uncertainty in object location.
- Updates frame by frame.

Weaknesses:

- Can drift if bad samples are used for training.
- Less robust than modern deep trackers.

---

## 52. BOOSTING Tracker

The BOOSTING tracker uses online AdaBoost.

It trains multiple weak classifiers and combines them into a strong classifier.

Process:

1. Sample positive and negative patches in each frame.
2. Train weak classifiers.
3. Focus learning on hard-to-classify patches.
4. Combine weak classifiers into a strong decision function.
5. Update online during tracking.

Strength:

- Adapts to changing object appearance.

Weakness:

- Sensitive to drift when trained on bad samples.

---

## 53. TLD Tracker

TLD means Tracking-Learning-Detection.

It has three components:

1. Tracking.
2. Learning.
3. Detection.

### 53.1 Tracking

Short-term component.

Handles frame-to-frame motion.

### 53.2 Detection

Long-term component.

Scans the image for object-like regions.

Can reinitialize the tracker if tracking fails.

### 53.3 Learning

Observes predictions from the tracker and detector.

Updates the detector over time.

Strength:

- Designed for long-term tracking and recovery.

Weakness:

- More complex.
- Can still drift if learning is wrong.

---

# Part VI: Correlation-Filter-Based Trackers

## 54. What Are Correlation Filter Trackers?

Correlation-filter-based trackers learn a filter that produces a strong response at the target location.

Motion model:

Search in a window around the previous location.

Appearance model:

Learn a correlation filter and update it online.

Key property:

They work efficiently in the frequency domain.

This enables real-time performance.

Strengths:

- Fast.
- Online adaptation.
- Good for real-time tracking.

Weaknesses:

- Not semantic.
- May drift under occlusion or severe appearance change.

---

## 55. MOSSE Filter

MOSSE learns a grayscale filter from the initial target appearance.

It trains by minimizing squared error in the response map.

The filter is updated online using exponential moving average.

Strengths:

- Extremely fast.
- Can run at hundreds of FPS on CPU.
- Very low latency.

Weaknesses:

- Sensitive to scale changes.
- Sensitive to rotation.
- Sensitive to large appearance changes.

---

## 56. KCF Tracker

KCF means Kernelized Correlation Filter.

It extends MOSSE by applying nonlinear kernels in feature space.

It also learns filters over multi-channel descriptors such as HOG.

Optimization remains efficient because it uses FFT-based computation.

Strengths:

- Better accuracy than MOSSE.
- More robust to appearance changes.
- Efficient.

Weaknesses:

- Fixed window size.
- No scale adaptation.
- Can drift when the object leaves the search window.
- Can fail under occlusion.

---

## 57. CSRT Tracker

CSRT uses channel and spatial reliability.

It down-weights unreliable background regions and focuses more on reliable target regions.

Strengths:

- More robust than MOSSE and KCF.
- Handles scale changes better.
- Handles rotation and clutter better.
- More robust under partial occlusion.

Weaknesses:

- Slower.
- More computationally heavy.
- Not necessarily robust to complete occlusion.

---

## 58. Comparison of Correlation Filter Trackers

| Tracker | Key Features                    | Strengths                          | Weaknesses                            |
| ------- | ------------------------------- | ---------------------------------- | ------------------------------------- |
| MOSSE   | Grayscale-only, adaptive filter | Very fast, low latency             | Sensitive to scale and rotation       |
| KCF     | Kernel trick, fixed-size window | Better accuracy than MOSSE         | No scale adaptation                   |
| CSRT    | Channel and spatial reliability | Robust to scale, rotation, clutter | Slower and more computationally heavy |

---

# Part VII: Deep-Learning-Based Tracking

## 59. Main Types of Deep Trackers

Deep-learning-based tracking methods include:

- Regression-based trackers.
- Siamese-network trackers.
- Transformer-based trackers.
- Hybrid and online learning trackers.

Examples:

Regression-based:

- GOTURN.
- TrackerNano.

Siamese:

- SiamFC.
- SiamRPN.
- SiamRPN++.
- SiamMask.

Transformer-based:

- TransTrack.
- STARK.
- TransT.

Hybrid:

- MDNet.
- ATOM.
- DiMP.

---

## 60. Regression-Based Trackers

Regression-based trackers predict bounding boxes directly.

They treat tracking as a prediction problem.

Input:

- Target crop from previous frame.
- Search crop from current frame.

Output:

- Predicted bounding box in the current frame.

Strengths:

- Simple.
- Fast.
- Good for real-time tracking.

Weaknesses:

- Struggle with long-term consistency.
- Can fail under drastic appearance changes.
- Often lack online adaptation.

---

## 61. GOTURN

GOTURN is a regression-based tracker.

Input:

- Target crop from previous frame.
- Search crop from current frame.

Output:

- Predicted bounding box.

Training datasets:

- ImageNet VID.
- ALOV300++.

Properties:

- First real-time DNN tracker.
- Runs at over 100 FPS.
- No online update.

Strengths:

- Very fast.
- Simple.

Weaknesses:

- Not robust to drastic appearance changes.
- May fail under occlusion.
- Does not adapt online.

---

## 62. TrackerNano

TrackerNano is designed for ultra-low-latency tracking.

Architecture:

- Compact CNN backbone.
- Input: object patch at t - 1 plus search image at t.

Designed for:

- ARM devices.
- Jetson devices.
- Embedded platforms.

Properties:

- Low power usage: approximately 1 to 5 W.
- Speed: 30 to 60 FPS on embedded GPUs.
- Good trade-off between speed and acceptable accuracy.

---

## 63. Siamese-Network Trackers

Siamese trackers match objects across frames.

They use two inputs:

1. Template from the first frame.
2. Search region from the current frame.

Both inputs are passed through the same CNN feature extractor.

Then cross-correlation is used to find the best match.

Basic idea:

The object in the current frame should look similar to the template.

---

## 64. SiamFC

SiamFC is a Siamese fully-convolutional tracker.

Input:

- Template from frame 0.
- Search region from current frame.

It learns a similarity function using offline training.

Properties:

- No online update.
- No region proposals.
- Simple design.
- Runs at about 86 FPS.
- Precision around 77% on OTB-2015.

Strengths:

- Fast.
- Simple.
- Learns matching from data.

Weaknesses:

- No online adaptation.
- Limited scale handling compared to later Siamese trackers.

---

## 65. SiamRPN

SiamRPN extends SiamFC by adding a Region Proposal Network.

It learns:

- Classification.
- Bounding box regression.

Advantages:

- Better localization.
- Better scale estimation.
- Faster and more accurate than earlier Siamese trackers.

Properties:

- Runs at about 160 FPS.
- Strong performance on OTB and VOT 2016/2017.

---

## 66. Transformer-Based Trackers

Transformer-based trackers use attention mechanisms.

They model relationships between patches in:

- Template image.
- Search image.

Attention helps the model compare object and search regions more flexibly.

Strengths:

- More robust to drastic changes.
- More robust under occlusion.
- Better global reasoning.

Weaknesses:

- Slower than lightweight trackers.
- More computationally expensive.

---

## 67. TransT

TransT is a transformer-based tracker.

Architecture:

- ResNet-50 backbone.
- Fusion module with spatial attention and channel attention.
- No anchor boxes required.

Properties:

- Real-time capable.
- About 25 to 30 FPS on a single GPU.
- LaSOT AUC around 69.1.
- TrackingNet AUC around 81.4.

---

## 68. STARK

STARK is a fully transformer-based tracker.

Properties:

- End-to-end design.
- Uses spatial and temporal encoding.
- Does not require a separate localization step.
- Real-time speed around 30 FPS.

Reported performance:

- Success AUC 67.1 on LaSOT.
- Success AUC 68.8 on GOT-10k.

Strengths:

- Strong global modeling.
- Robust under challenging tracking conditions.

---

## 69. Hybrid and Online Learning Trackers

Hybrid trackers combine deep features with online learning.

Examples:

- MDNet.
- ATOM.
- DiMP.

Strengths:

- Strong under challenging conditions.
- Can adapt during tracking.
- Use both target and background information.

Weaknesses:

- More computationally heavy.
- More complex.

---

## 70. MDNet

MDNet fine-tunes during tracking.

It uses deep features and online adaptation.

Strength:

- Strong appearance adaptation.

Weakness:

- Computationally expensive.
- Slower than many real-time trackers.

---

## 71. ATOM and DiMP

ATOM and DiMP learn both:

- What to track.
- Where it is.

DiMP learns from both target and background.

Properties of DiMP:

- Fully differentiable.
- End-to-end training.
- No handcrafted loss or manual model update rules.

Strength:

- Strong accuracy and robustness.

Weakness:

- More computationally expensive than classical trackers.

---

# Part VIII: Benchmark Datasets

## 72. Object Tracking Benchmark Datasets

| Dataset       |  Clips | Notes                                                                       |
| ------------- | -----: | --------------------------------------------------------------------------- |
| OTB-2015      |    100 | Short and low resolution, low object variety, center-framed targets         |
| VOT 2016-2021 | 60-100 | Short clips, high quality, object diversity                                 |
| LaSOT         |  1400+ | Long videos, high resolution, challenging objects, diversity, 70 categories |
| GOT-10k       |    10k | Real-world clips, 563 categories, unseen test set                           |
| TrackingNet   |    30k | Real-world YouTube videos, high diversity                                   |

---

## 73. OTB-2015

OTB-2015 is an older tracking benchmark.

Properties:

- 100 videos.
- Short clips.
- Low resolution.
- Low object variety.
- Often center-framed targets.

Useful for quick comparison, but less representative of modern real-world tracking.

---

## 74. VOT

VOT is the Visual Object Tracking challenge.

Properties:

- 60 to 100 videos depending on the year.
- Short clips.
- High-quality annotation.
- Good object diversity.

VOT commonly uses metrics such as Expected Average Overlap.

---

## 75. LaSOT

LaSOT is a large-scale single-object tracking dataset.

Properties:

- More than 1400 videos.
- Long videos.
- High resolution.
- Challenging objects.
- Diverse scenes.
- 70 object categories.

Useful for evaluating long-term tracking robustness.

---

## 76. GOT-10k

GOT-10k is a large-scale real-world tracking dataset.

Properties:

- Around 10,000 clips.
- 563 object categories.
- Unseen test set.

Important because the test categories are not seen during training.

This evaluates generalization.

---

## 77. TrackingNet

TrackingNet is a large-scale tracking dataset based on YouTube videos.

Properties:

- Around 30,000 videos.
- Real-world videos.
- High diversity.

Useful for evaluating tracking in realistic conditions.

---

# Part IX: Evaluation Metrics

## 78. Precision

Precision measures the percentage of frames where the predicted object center is within a fixed pixel threshold of the ground-truth center.

Example threshold:

20 pixels.

If the predicted center is close enough to the true center, the frame is counted as correct.

Strength:

- Simple and intuitive.

Weakness:

- Sensitive to object size.
- A 20-pixel error may be small for a large object but large for a small object.

---

## 79. Success / IoU

Success measures whether the predicted bounding box overlaps sufficiently with the ground-truth bounding box.

It is based on IoU.

IoU means Intersection over Union.

Formula:

IoU = Area of intersection / Area of union

A frame is successful if IoU is above a selected threshold.

---

## 80. AUC

AUC means Area Under Curve.

In tracking, it usually refers to the area under the success plot.

It measures average tracking accuracy across IoU thresholds.

Strength:

- More complete than using one fixed IoU threshold.
- Common in tracking benchmarks.

Weakness:

- Does not explicitly model tracker resets or failure recovery.

---

## 81. EAO

EAO means Expected Average Overlap.

It combines:

- Accuracy.
- Failure rate.

It is used in the VOT challenge.

EAO is useful because it penalizes trackers that fail often and require resets.

This makes it more suitable for scenarios where robustness matters.

Example:

A tracker that is accurate for a few frames but frequently loses the object may have good short-term IoU but poor EAO.

---

# Part X: Tracking Frameworks

## 82. OpenCV

OpenCV provides classical tracking tools.

Supported trackers include:

- MIL.
- BOOSTING.
- KCF.
- TLD.
- CSRT.

OpenCV also provides:

- cv2.Tracker API.
- Lightweight real-time tracking.
- Single-object tracking tools.

Strength:

- Easy to use.
- Good for classical trackers.
- Useful for practical experiments.

---

## 83. MMTracking

MMTracking is part of the MMDetection ecosystem.

It provides unified support for:

- SOT: Single Object Tracking.
- MOT: Multiple Object Tracking.

Properties:

- Modular design.
- Supports Siamese trackers.
- Supports transformer trackers.
- Useful for research and benchmarking.

---

## 84. PyTracking

PyTracking is a modular framework for state-of-the-art deep trackers.

It is built on PyTorch.

Supported trackers include:

- DiMP.
- ATOM.
- STARK.
- TransT.
- KYS.
- PrDiMP.

Useful for:

- Research.
- Reproducing modern trackers.
- Comparing deep tracking models.

---

# Part XI: Key Comparisons

## 85. Motion Estimation Methods

| Method                 | Main Idea                                   | Strengths                                   | Weaknesses                                    |
| ---------------------- | ------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| Lucas-Kanade           | Local patch motion                          | Fast, simple, good for small motion         | Fails with large motion and motion boundaries |
| Horn-Schunck           | Global smooth flow                          | Dense and smooth optical flow               | Oversmooths boundaries                        |
| Coarse-to-fine         | Estimate motion from low to high resolution | Handles larger motion                       | Can miss small details                        |
| FlowNet                | CNN predicts flow directly                  | End-to-end learning                         | Early versions less accurate                  |
| PWC-Net                | Pyramid, warping, cost volume               | Efficient and inspired by classical methods | Still depends on good feature matching        |
| RAFT                   | All-pairs cost volume and recurrent updates | Very accurate                               | High memory cost                              |
| Transformer-based flow | Attention-based matching                    | Strong global reasoning                     | Computationally heavy                         |

---

## 86. Object Tracking Methods

| Method Type          | Examples                       | Strengths                      | Weaknesses                      |
| -------------------- | ------------------------------ | ------------------------------ | ------------------------------- |
| Motion model         | Kalman filter, particle filter | Fast, handles noisy motion     | Weak appearance understanding   |
| Appearance-based     | Mean Shift, CamShift           | Simple and efficient           | Sensitive to appearance changes |
| Optical flow-based   | KLT                            | Fast point tracking            | Sparse, fails with large motion |
| Online learning      | MIL, BOOSTING, TLD             | Adapts to target changes       | Can drift                       |
| Correlation filters  | MOSSE, KCF, CSRT               | Fast, practical                | Limited semantic understanding  |
| Regression-based DNN | GOTURN, TrackerNano            | Very fast                      | Long-term consistency issues    |
| Siamese trackers     | SiamFC, SiamRPN                | Strong matching, fast          | Limited online adaptation       |
| Transformer trackers | TransT, STARK                  | Robust, global reasoning       | More expensive                  |
| Hybrid trackers      | MDNet, ATOM, DiMP              | Strong accuracy and adaptation | Computationally heavier         |

---

# Part XII: Key Takeaways

## 87. Motion Estimation Takeaways

Motion estimation is central to video understanding.

It helps with:

- Object tracking.
- Video compression.
- Video stabilization.
- Frame interpolation.
- Human motion analysis.

Classical optical flow is based on brightness constancy and smoothness assumptions.

The optical flow constraint equation is:

I_x u + I_y v + I_t = 0

But this equation is underdetermined, which creates the aperture problem.

Lucas-Kanade solves this locally by assuming patch-level motion.

Horn-Schunck solves it globally by assuming neighboring pixels have similar motion.

Deep learning methods such as FlowNet, PWC-Net, and RAFT learn optical flow from data and often combine classical concepts with neural networks.

RAFT is powerful but has major memory limitations because of all-pairs cost volume.

---

## 88. Object Tracking Takeaways

Object tracking estimates the position of a target object over time.

Tracking is not the same as detection.

Tracking uses:

- Temporal continuity.
- Motion models.
- Appearance models.
- Online adaptation.

Traditional trackers are fast and simple but less robust.

Deep trackers are more robust but often require more computation.

Correlation filters are a strong practical baseline for real-time single-object tracking.

Siamese trackers are fast and effective because they formulate tracking as template matching.

Transformer trackers improve robustness using attention but are more expensive.

Evaluation should consider both accuracy and robustness.

Important metrics:

- Precision.
- Success / IoU.
- AUC.
- EAO.

EAO is especially useful when tracker failure and resets matter.

# Lesson 04: 3D Vision

## 1. Objective

This lecture focuses on two main problems in 3D vision:

1. Predicting 3D shapes from a single image.
2. Processing 3D input data.

Example tasks:

- Input: RGB image of a chair.
- Output: 3D shape of the chair.
- Input: 3D shape.
- Output: object class, for example “chair”.

The lecture also mentions that 3D vision is a broad field including:

- 3D representations.
- Computing correspondences.
- Multi-view stereo.
- Structure from motion.
- 3D object pose estimation.
- SLAM.
- Differentiable graphics.
- 3D sensors.
- Simulation environments.

---

# Part I: Multi-View CNN

## 2. Multi-View CNN: The 2D Hack

A Multi-View CNN processes a 3D object by rendering or capturing it from multiple 2D views.

Instead of directly processing the 3D shape, the method uses standard 2D CNNs on multiple images.

Input:

N views of the same object

Tensor shape:

[N, C, H, W]

where:

- N = number of camera views.
- C = image channels.
- H = image height.
- W = image width.

The main question is:

How do we combine the N different camera views into one object descriptor?

---

## 3. Multi-View CNN Tensor Pipeline

The pipeline:

1. Capture N images of the object.
2. Extract 2D features from each image using a CNN.
3. Combine view features using view pooling.
4. Produce one final object descriptor.

View pooling:

- Usually element-wise max-pooling across all views.
- Keeps the strongest feature response among all camera views.

Final object descriptor:

[1, F]

where:

- F = descriptor dimension.

This descriptor can then be used for classification or retrieval.

---

## 4. Multi-View CNN Architecture

A common Multi-View CNN structure:

1. CNN1 extracts features from each view.
2. View pooling combines features across views.
3. CNN2 or additional layers produce a final shape descriptor.
4. A softmax layer predicts the object class.

Example output classes:

- bathtub.
- bed.
- chair.
- desk.
- dresser.
- toilet.

The main advantage is that the model can reuse powerful 2D CNN architectures.

The main limitation is that this is not a truly native 3D representation.

It treats 3D shape recognition as a set of 2D image recognition problems.

---

## 5. Multi-View CNN Performance

The lecture compares several methods for 3D shape classification and retrieval.

Approximate results:

| Method                   | Classification Accuracy | Retrieval mAP |
| ------------------------ | ----------------------: | ------------: |
| SPH                      |                   68.2% |         33.3% |
| LFD                      |                   75.5% |         40.9% |
| 3D ShapeNets             |                   77.3% |         49.2% |
| FV, 12 views             |                   84.8% |         43.9% |
| CNN, 12 views            |                   88.6% |         62.8% |
| MVCNN, 12 views          |                   89.9% |         70.1% |
| MVCNN + metric, 12 views |                   89.5% |         80.2% |
| MVCNN, 80 views          |                   90.1% |         70.4% |
| MVCNN + metric, 80 views |                   90.1% |         79.5% |

Key idea:

Multi-view CNNs can perform very well because 2D CNNs are strong and mature.

---

# Part II: 2.5D Representations

## 6. What Are 2.5D Representations?

2.5D representations add partial 3D information to a 2D image.

They do not fully represent the complete 3D shape.

Common 2.5D representations:

- Depth maps.
- Surface normals.

They are called 2.5D because they give 3D-related information from a particular viewpoint.

---

## 7. Depth Maps

A depth map stores the distance from the camera to the object or scene at each pixel.

RGB image shape:

3 x H x W

Depth map shape:

H x W

or sometimes:

1 x H x W

Each pixel value represents depth.

RGB image + depth image = RGB-D image.

RGB-D data can be captured directly by some 3D sensors, for example:

- Intel RealSense.
- Microsoft Kinect.

---

## 8. Predicting Depth Maps

A neural network can predict depth from a single RGB image.

Input:

RGB image:

3 x H x W

Model:

Fully convolutional network

Output:

Predicted depth image:

1 x H x W

Training:

The model compares predicted depth with ground-truth depth using a per-pixel loss.

A basic loss can be L2 distance.

---

## 9. Scale and Depth Ambiguity

Depth prediction from a single image has an important ambiguity:

A small close object can look exactly the same as a larger far-away object.

This creates scale ambiguity.

Problem:

From a single image, absolute scale and absolute depth are ambiguous.

The model must infer depth from visual cues such as:

- Perspective.
- Object size priors.
- Texture.
- Shadows.
- Occlusion.
- Scene context.

---

## 10. Scale-Invariant Depth Loss

Because absolute depth is ambiguous, depth prediction often uses a scale-invariant loss.

The goal is to compare predicted and ground-truth depth while being less sensitive to global scale differences.

Instead of only penalizing absolute depth error, the loss focuses more on relative depth structure.

This is useful because monocular depth estimation often cannot recover exact metric scale.

---

## 11. Surface Normals

Surface normals describe the orientation of the surface at each pixel.

For each pixel, a surface normal gives a vector perpendicular to the object surface in 3D space.

RGB image shape:

3 x H x W

Surface normals shape:

3 x H x W

Each normal has three components:

- x component.
- y component.
- z component.

Surface normals are useful because they describe local geometry.

---

## 12. Predicting Surface Normals

A neural network can predict surface normals from an RGB image.

Input:

RGB image:

3 x H x W

Model:

Fully convolutional network

Output:

Predicted normals:

3 x H x W

Ground truth:

3 x H x W

Loss:

A common loss compares the angle between predicted and ground-truth normal vectors.

Cosine similarity:

(x · y) / (|x| |y|)

Recall:

x · y = |x| |y| cos(theta)

If the predicted vector has the same direction as the ground-truth vector, the cosine similarity is high.

---

# Part III: 3D Representations

## 13. Main 3D Representations

The lecture presents several common 3D shape representations:

- Voxel grid.
- Point cloud.
- Triangle mesh.
- Implicit surface or implicit function.

Each representation has different trade-offs.

---

# Part IV: Voxels

## 14. Voxel Grids

A voxel grid represents a 3D shape as a regular 3D grid of occupancies.

Shape:

V x V x V

Each voxel stores whether that small cube of space is occupied.

A voxel grid is similar to a segmentation mask, but in 3D.

Example:

A 2D mask tells which image pixels belong to an object.

A voxel grid tells which 3D cells belong to an object.

---

## 15. Advantages and Disadvantages of Voxels

Advantages:

- Conceptually simple.
- Easy to understand.
- Compatible with 3D convolutions.
- Similar to image tensors extended into 3D.

Disadvantages:

- Requires high spatial resolution to capture fine structures.
- Memory usage grows cubically with resolution.
- Scaling to high resolution is difficult.

Memory complexity:

V x V x V

If V doubles, memory increases by 8 times.

---

## 16. Processing Voxel Inputs with 3D Convolution

Voxel grids can be processed using 3D CNNs.

Example input:

1 x 30 x 30 x 30

This means:

- 1 input channel.
- 30 x 30 x 30 voxel grid.

The network applies 3D convolution layers.

Example output stages from the lecture:

- Output: 48 x 13 x 13 x 13.
- Output: 160 x 5 x 5 x 5.
- Output: 512 x 2 x 2 x 2.
- Fully connected layer.
- Class scores.

This is analogous to 2D CNN classification, but all operations are 3D.

---

## 17. Generating Voxel Shapes from Images

A model can generate voxels from a single RGB image.

Pipeline:

1. Input image:

3 x 112 x 112

2. 2D CNN extracts image features:

C x H x W

3. Convert 2D features into 3D features:

C' x D' x H' x W'

4. 3D CNN predicts voxel grid:

1 x V x V x V

Main challenge:

How do we transition from 2D features to a 3D volume?

---

## 18. Transitioning from 2D to 3D

The lecture presents several strategies.

### 18.1 Reshaping 2D Features into a 3D Volume

The 2D feature tensor is reshaped into a 3D feature volume.

Requirement:

The total number of elements must match.

Example:

2D feature map:

512 x 16 x 16 = 131,072 elements

3D volume:

64 x 8 x 16 x 16 = 131,072 elements

Disadvantage:

The reshaped dimensions have no clear geometric meaning.

---

### 18.2 Depth Replication

The 2D feature map is copied D times along the depth axis.

Example:

2D feature map:

256 x 28 x 28

3D volume:

256 x 8 x 28 x 28

Advantage:

- Simple and fast.

Disadvantage:

- It is a quick hack.
- It does not truly infer depth structure.

---

### 18.3 Learned Depth Projection

A network learns to project 2D features into a 3D volume.

Idea:

- Use discrete depth bins.
- For each pixel in the 2D feature map, predict features at different depths.

Common in:

- Multi-view stereo.
- Monocular depth estimation.

Advantage:

- More geometrically meaningful than reshaping or replication.

---

### 18.4 MLP Projection

The 2D feature map is flattened.

Then an MLP projects it into a 3D latent space.

Finally, the output is reshaped into a 3D volume.

Common in:

- Autoencoder-based shape generators.

Advantage:

- Flexible.

Disadvantage:

- Can lose spatial structure.
- Can be memory-heavy.

---

## 19. Voxel Memory Problem

Voxel memory grows very quickly.

A 1024 x 1024 x 1024 voxel grid with float32 values requires about 4 GB of memory.

Reason:

1024^3 = 1,073,741,824 values

Each float32 value uses 4 bytes.

Memory:

1,073,741,824 x 4 bytes ≈ 4 GB

This is only for one voxel grid.

It does not include:

- Batch dimension.
- Feature channels.
- Gradients.
- Activations.
- Model parameters.

Therefore, dense high-resolution voxels are expensive.

---

## 20. Scaling Voxels with Octrees

Octrees reduce memory usage by representing space adaptively.

Instead of storing every voxel at the same resolution, an octree subdivides space only where detail is needed.

Dense voxel grid:

- Same resolution everywhere.
- Wastes memory in empty regions.

Octree:

- Coarse representation in empty or simple regions.
- Fine representation near surfaces and detailed structures.

This helps scale voxel representations to higher effective resolution.

---

# Part V: Point Clouds

## 21. Point Clouds

A point cloud represents a 3D shape as a set of P points in 3D space.

Shape:

P x 3

Each point has coordinates:

(x, y, z)

Point clouds can also include additional information such as RGB color:

(x, y, z, r, g, b)

---

## 22. Advantages and Disadvantages of Point Clouds

Advantages:

- Can represent fine structures without huge dense grids.
- More memory-efficient than voxels.
- Natural output from many 3D sensors.
- Simple geometric representation.

Disadvantages:

- Does not explicitly represent surfaces.
- Does not directly store mesh connectivity.
- Extracting a mesh requires post-processing.
- Requires specialized architectures and losses.
- Points are unordered, so the model must be permutation-invariant.

---

# Part VI: PointNet

## 23. PointNet

PointNet is a neural network architecture for point clouds.

Input:

Point cloud:

P x 3

The basic PointNet pipeline:

1. Run an MLP on each point.
2. Produce point features:

P x D

3. Apply max-pooling over points.
4. Produce a global point cloud descriptor:

D

5. Use fully connected layers.
6. Predict class scores:

C

---

## 24. Why Use 1D Convolution Instead of Separate MLP Calls?

Running an independent MLP on each point one by one would bottleneck the GPU.

Instead, PointNet treats points as a 1D spatial array.

It uses 1D convolution with kernel size 1.

This is equivalent to applying the same MLP to every point, but it is much more efficient.

Advantages:

- Massive parallelism.
- Hardware-optimized computation.
- Same weights applied to all points.

---

## 25. Why Max-Pooling?

Point clouds are unordered sets.

The result should not depend on the order of points.

Max-pooling gives permutation invariance.

Mean pooling would average information across all points.

Max-pooling selects the strongest activation for each feature dimension.

This allows the network to preserve critical point evidence.

---

## 26. Critical Points in PointNet

PointNet uses max-pooling to distill the most important information from the point cloud.

Only some points may determine the global descriptor.

These are called critical points.

Example:

For an airplane, critical points may be:

- Wing tips.
- Nose.
- Tail.
- Engine regions.

The global descriptor is built from the strongest feature responses over the point set.

---

## 27. PointNet for 3D Object Classification

For classification:

Input:

Point cloud P x 3

Process:

1. Per-point feature extraction.
2. Global max-pooling.
3. Fully connected classifier.

Output:

Object class.

Examples:

- airplane.
- mug.
- table.
- car.

---

## 28. PointNet for 3D Object Segmentation

For segmentation, the model must predict a label for each point.

Pipeline:

1. Extract local point features.
2. Extract global feature using max-pooling.
3. Copy and concatenate the global feature with each local point feature.
4. Run per-point classification.

Output:

A class label for every point.

This is used for:

- Semantic segmentation.
- Object part segmentation.

---

## 29. Locality vs Identity Problem

For segmentation, a point’s local features may not be enough.

Example:

A chair leg and a table leg may look locally similar.

To classify a part correctly, the network needs:

- Local feature information.
- Global object context.

PointNet solves this by concatenating:

- Local feature vector.
- Global feature vector.

The global vector tells the model what object the point belongs to.

---

## 30. PointNet Applications

PointNet can be used for:

- 3D object classification.
- Semantic segmentation on point clouds.
- Object part segmentation.
- Scene understanding.
- Indoor point cloud analysis.

Examples:

Semantic segmentation:

- floor.
- wall.
- chair.
- table.
- ceiling.

Object part segmentation:

- airplane wing.
- airplane body.
- chair leg.
- chair back.
- mug handle.

---

# Part VII: Sensor Fusion

## 31. Point Clouds with RGB

Point clouds can store only geometry:

(x, y, z)

or geometry plus color:

(x, y, z, r, g, b)

Adding RGB helps because geometry alone may not distinguish objects with similar shape.

Example:

A point cloud with RGB can include both:

- 3D structure.
- Texture or appearance information.

---

## 32. Dense Fusion

Dense fusion combines point cloud features with RGB image features.

Pipeline:

1. Point cloud branch:

PointNet extracts per-point features.

2. RGB branch:

A CNN extracts per-pixel image features.

3. Feature association:

Each 3D point is associated with an image pixel.

4. Per-point feature fusion:

Point features and pixel features are combined.

5. Final prediction:

Classification, segmentation, or another task.

Dense fusion is useful when both RGB and depth/point cloud data are available.

---

# Part VIII: Generating Point Clouds

## 33. Generating Point Clouds from Images

A model can generate a point cloud from a single RGB image.

Input:

3 x H x W

A 2D CNN extracts image features:

C x H' x W'

The network can have two branches:

### 33.1 Fully Connected Branch

Predicts a global set of points:

P1 x 3

### 33.2 Convolutional Branch

Predicts local point sets for each spatial location:

(P2 x 3) x H' x W'

Final point cloud:

(P1 + H'W'P2) x 3

This allows the model to combine global shape prediction with local detail prediction.

---

## 34. Loss Function for Point Clouds

To train a point cloud generator, we need a differentiable way to compare two point clouds as sets.

The lecture introduces Chamfer distance.

---

## 35. Chamfer Distance

Chamfer distance compares two point sets by nearest-neighbor distances.

For each point in predicted point cloud:

- Find the nearest point in the ground-truth point cloud.
- Add the squared L2 distance.

For each point in the ground-truth point cloud:

- Find the nearest point in the predicted point cloud.
- Add the squared L2 distance.

Chamfer distance is symmetric when both directions are included.

Intuition:

A good predicted point cloud should be close to the ground truth, and the ground truth should be covered by predicted points.

Advantages:

- Differentiable almost everywhere.
- Works with unordered point sets.
- Does not require point-to-point correspondence.

Weaknesses:

- Can tolerate uneven point density.
- May not preserve surface structure perfectly.

---

# Part IX: Triangle Meshes

## 36. Triangle Mesh

A triangle mesh represents a 3D shape as a set of triangles.

It consists of:

- Vertices.
- Faces.

Vertices:

A set of V points in 3D space.

Faces:

A set of triangles over the vertices.

Each face connects three vertices.

---

## 37. Advantages and Disadvantages of Triangle Meshes

Advantages:

- Standard representation in computer graphics.
- Explicitly represents 3D shapes.
- Efficient for flat surfaces.
- Adaptive: can allocate more faces to detailed regions.
- Can attach data to vertices and interpolate over the surface.

Attached data can include:

- RGB colors.
- Texture coordinates.
- Normal vectors.

Disadvantages:

- Nontrivial to process with neural networks.
- Mesh topology can vary.
- Loss functions are harder than for images.
- Different meshes can represent the same shape.

---

# Part X: Pixel2Mesh

## 38. Pixel2Mesh

Pixel2Mesh predicts a triangle mesh from a single RGB image.

Input:

Single RGB image of an object.

Output:

Triangle mesh for the object.

Key ideas:

1. Iterative mesh refinement.
2. Graph convolution.
3. Vertex-aligned features.
4. Chamfer loss function.

---

## 39. Pixel2Mesh: Iterative Mesh Refinement

Pixel2Mesh starts from an initial ellipsoid mesh.

The network predicts offsets for each vertex.

The mesh is refined step by step.

Process:

1. Start with simple initial mesh.
2. Predict vertex offsets.
3. Update vertex positions.
4. Repeat refinement.

This gradually deforms the initial mesh toward the target object shape.

---

## 40. Pixel2Mesh: Graph Convolution

A mesh can be treated as a graph.

- Vertices are graph nodes.
- Edges connect neighboring vertices.

Graph convolution updates each vertex feature using information from neighboring vertices.

Input:

Graph with feature vector at each vertex.

Output:

New feature vector for each vertex.

For vertex v_i with feature f_i, the new feature f'\_i depends on features of neighboring vertices N(i).

The same weights are used to compute outputs for all vertices.

This allows the network to process arbitrary mesh connectivity.

---

## 41. Pixel2Mesh: Vertex-Aligned Features

Problem:

How do we incorporate image features into mesh prediction?

Solution:

For each mesh vertex:

1. Use camera information to project the 3D vertex onto the image plane.
2. Use bilinear interpolation to sample a CNN feature at that projected 2D location.
3. Attach the sampled image feature to the vertex.

This gives every vertex access to relevant image information.

---

## 42. Pixel2Mesh: Mesh Loss

The same shape can be represented with different meshes.

Therefore, direct vertex-to-vertex comparison is not always valid.

Idea:

Convert meshes into point clouds and compute a point cloud loss.

Process:

1. Sample points from the surface of the ground-truth mesh offline.
2. Sample points from the predicted mesh online.
3. Compute Chamfer distance.

The loss measures geometric similarity without requiring identical mesh topology.

---

# Part XI: Mesh R-CNN

## 43. Mesh R-CNN

Mesh R-CNN extends Mask R-CNN from 2D object understanding to 3D mesh prediction.

Mask R-CNN:

2D image -> 2D shapes

Mesh R-CNN:

2D image -> triangle meshes

Input:

Single RGB image.

Output:

A set of detected objects.

For each object, Mesh R-CNN predicts:

- Bounding box.
- Category label.
- Instance segmentation.
- 3D triangle mesh.

---

## 44. Mesh R-CNN Pipeline

The pipeline:

1. Input image.
2. 2D object recognition.
3. 3D object voxel prediction.
4. 3D object mesh prediction.

Mesh R-CNN uses a mesh head in addition to the normal detection and segmentation components.

---

## 45. Mesh R-CNN Topology Problem

Mesh deformation can produce good results.

However, topology is fixed by the initial mesh.

This includes:

- Number of vertices.
- Faces.
- Genus.
- Connected components.

Problem:

If the initial mesh topology is wrong, deformation alone may not recover the correct shape.

Solution in Mesh R-CNN:

Use voxel predictions to create the initial mesh prediction.

This gives a better starting topology before mesh refinement.

---

# Part XII: Implicit Functions

## 46. Implicit Functions

An implicit function represents a 3D shape by classifying arbitrary 3D points as inside or outside the shape.

The model learns a function:

o(x)

where:

- x is a 3D point.
- o(x) is the occupancy probability or signed value.

The object surface is represented as a level set.

For occupancy:

surface = {x : o(x) = 1/2}

This means the surface is where the function changes from outside to inside.

---

## 47. Explicit vs Implicit Shape

Explicit representations directly store the shape.

Examples:

- Mesh.
- Voxel grid.
- Point cloud.

Implicit representations store a function that defines the shape.

Advantages:

- Can represent continuous surfaces.
- Not limited to a fixed grid resolution.
- Can represent complex topology.

Disadvantages:

- Rendering or extracting surfaces may require sampling.
- Training and inference can be slow.
- Harder to visualize directly.

---

## 48. Algebraic Surfaces

An algebraic surface is an implicit surface defined as the zero set of a polynomial in x, y, z.

Example idea:

f(x, y, z) = 0

The surface is all points where the function equals zero.

Problem:

Simple polynomials cannot easily represent very complex shapes.

---

## 49. Constructive Solid Geometry

Constructive solid geometry combines implicit geometry using Boolean operations.

Operations:

- Union.
- Intersection.
- Difference.

This allows complex objects to be built from simpler shapes.

---

## 50. Level Set Methods

Implicit surfaces have useful properties.

They can naturally handle:

- Merging.
- Splitting.
- Topology changes.

However, it is hard to describe complex shapes in closed form.

Alternative:

Store a grid of values approximating the implicit function.

The surface is found where interpolated values equal zero.

This gives more explicit control over the shape, similar to a texture.

---

## 51. DeepSDF

DeepSDF is a neural implicit representation.

It learns a signed distance function.

For a 3D point x, the network predicts the signed distance to the nearest surface.

Interpretation:

- Negative value: inside the object.
- Positive value: outside the object.
- Zero: on the surface.

The surface is the zero level set.

DeepSDF can represent continuous shapes using a neural network.

---

# Part XIII: NeRF

## 52. Neural Radiance Fields

NeRF stands for Neural Radiance Fields.

A NeRF represents a 3D scene as a continuous function.

Input:

- 3D location:

(x, y, z)

- Viewing direction:

(theta, phi)

Output:

- Color:

(r, g, b)

- Density:

sigma

The model can synthesize novel views of a scene.

---

## 53. NeRF Input and Output

Input to the MLP:

(x, y, z, theta, phi)

Output:

(r, g, b, sigma)

where:

- r, g, b describe color.
- sigma describes volume density.

The density controls how much the point contributes to the rendered image along a camera ray.

---

## 54. Novel View Synthesis

NeRF is mainly used for novel view synthesis.

Given several images of a scene from different camera viewpoints, NeRF learns a continuous 3D representation.

After training, it can render the scene from new viewpoints.

This is useful for:

- 3D reconstruction.
- View synthesis.
- Virtual environments.
- Scene representation.

---

## 55. NeRF Variants

The lecture mentions several NeRF-related methods:

- Deformable NeRF.
- RawNeRF.
- BlockNeRF.

General idea:

Different NeRF variants improve NeRF for:

- Dynamic scenes.
- Raw camera data.
- Large-scale scenes.
- Outdoor reconstruction.
- Better quality or robustness.

---

## 56. Main Problem of Implicit Neural Representations

The main problem is speed.

NeRF-style methods are slow.

Training:

- Can take 1-2 days on a V100 GPU for just a single scene.

Inference:

To render one image:

256 x 256 pixels x 224 samples per pixel = 14.6 million MLP forward passes

This makes rendering expensive.

---

# Part XIV: 3D Gaussian Splatting

## 57. 3D Gaussian Splatting

3D Gaussian Splatting is a newer representation for scenes.

Instead of querying a continuous MLP along every ray, it represents the scene as a discrete set of 3D Gaussians.

Rendering blends Gaussians along the camera ray.

---

## 58. NeRF vs 3D Gaussian Splatting

NeRF:

- Queries a continuous MLP along the ray.
- Fitting is slow.
- Rendering is slow.
- Rendering can take around 10 seconds per frame at moderate resolution.

3D Gaussian Splatting:

- Blends a discrete set of Gaussians along the ray.
- Scene fitting is faster.
- Can fit a scene in a few minutes.
- Supports real-time rendering.
- Produces high-quality visuals.

---

## 59. Dynamic 3D Gaussians

Dynamic 3D Gaussians extend Gaussian splatting to dynamic scenes.

Goal:

Represent scenes where geometry or appearance changes over time.

This is useful for:

- Moving people.
- Dynamic environments.
- Video-based scene reconstruction.

---

## 60. 3D Gaussian Splatting for SLAM

3D Gaussian Splatting can also be connected to SLAM.

SLAM means Simultaneous Localization and Mapping.

Goal:

- Estimate camera motion.
- Build a map of the environment at the same time.

Using Gaussian representations can support:

- Real-time mapping.
- High-quality rendering.
- Dense scene reconstruction.

---

# Part XV: Datasets

## 61. ShapeNet

ShapeNet is a large-scale synthetic 3D object dataset.

Properties:

- Around 3 million models.
- Large-scale synthetic objects.
- ModelNet was absorbed by ShapeNet.
- ShapeNetCore contains 51.3K models in 55 categories.

Use cases:

- 3D object classification.
- Shape reconstruction.
- Mesh generation.
- Point cloud learning.

---

## 62. Pix3D

Pix3D contains real images paired with 3D shapes.

Properties:

- 10,069 images.
- 395 shapes.
- Objects include IKEA furniture and 3D scans.

Pix3D supports tasks such as:

- Predicting 3D shape from real images.
- Predicting many objects per scene.
- Mesh prediction.
- Amodal completion.

---

## 63. Amodal Completion

Amodal completion means predicting the full object shape, including occluded parts.

Example:

If a chair is partially hidden behind a table, the model still predicts the complete chair shape.

This requires the model to infer missing geometry from context and object priors.

---

## 64. Segmentation Failures Propagate to Meshes

In pipelines such as Mesh R-CNN, earlier 2D predictions affect later 3D predictions.

If the 2D segmentation fails, the predicted mesh may also be wrong.

This shows that 3D prediction pipelines can be sensitive to upstream errors.

---

## 65. PartNet

PartNet is a dataset for fine-grained part understanding.

Properties:

- Fine-grained parts.
- Includes mobility information.
- Instance-level annotations.
- Hierarchical structure.

Useful for:

- Part segmentation.
- Object structure understanding.
- Robotic manipulation.
- Fine-grained 3D reasoning.

---

# Part XVI: Simulation Environments

## 66. iGibson

iGibson is a simulation environment for embodied AI and 3D scene understanding.

It is useful for:

- Navigation.
- Robotics.
- Interaction with indoor scenes.
- Training agents in realistic environments.

---

## 67. BEHAVIOR-1K

BEHAVIOR-1K is a simulation benchmark for household activities and embodied AI.

It is designed for tasks involving:

- Object interaction.
- Human-like activities.
- Long-horizon planning.
- Robotic behavior in simulated environments.

---

# Part XVII: Production Ecosystem

## 68. 3D Vision Toolkit

A production 3D vision ecosystem usually includes tools for:

- 3D data loading.
- Mesh processing.
- Point cloud processing.
- Rendering.
- Simulation.
- Differentiable rendering.
- Dataset handling.
- Training neural networks on 3D data.

Common tool categories:

- Point cloud libraries.
- Mesh libraries.
- Differentiable rendering libraries.
- Simulation frameworks.
- Deep learning frameworks.

---

# Part XVIII: Comparison of 3D Representations

## 69. Representation Comparison

| Representation        | Main Idea                                   | Strengths                           | Weaknesses                              |
| --------------------- | ------------------------------------------- | ----------------------------------- | --------------------------------------- |
| Depth map             | Per-pixel distance from camera              | Simple, works with RGB-D sensors    | View-dependent, not full 3D             |
| Surface normals       | Per-pixel surface orientation               | Captures local geometry             | View-dependent, no complete shape       |
| Voxel grid            | 3D occupancy grid                           | Simple, works with 3D CNNs          | Huge memory cost                        |
| Point cloud           | Set of 3D points                            | Efficient, sensor-friendly          | No explicit surface                     |
| Triangle mesh         | Vertices and faces                          | Graphics standard, explicit surface | Harder for neural networks              |
| Implicit function     | Function defines inside/outside or distance | Continuous, flexible topology       | Slow sampling and rendering             |
| NeRF                  | Neural field for color and density          | High-quality novel views            | Slow training and inference             |
| 3D Gaussian Splatting | Discrete Gaussians in 3D space              | Fast rendering, high quality        | Newer ecosystem, scene-specific fitting |

---

## 70. Which Representation to Use?

Use depth maps when:

- The task is view-based.
- RGB-D sensor data is available.
- You need per-pixel geometry.

Use surface normals when:

- Local surface orientation matters.
- The task needs geometric cues but not full 3D shape.

Use voxels when:

- Simplicity is important.
- Resolution can be low.
- 3D CNNs are suitable.

Use point clouds when:

- Data comes from LiDAR or depth sensors.
- Memory efficiency matters.
- Exact surface connectivity is not required.

Use meshes when:

- Rendering or graphics integration is important.
- Explicit surfaces are needed.
- Geometry must be editable.

Use implicit functions when:

- Continuous surface representation is important.
- Topology may be complex.
- High geometric detail is required.

Use NeRF or Gaussian Splatting when:

- The goal is novel view synthesis.
- The scene is represented from multiple images.
- Visual quality matters.

---

# Part XIX: Key Takeaways

## 71. Main Ideas

3D vision extends computer vision from 2D images into 3D geometry.

Important tasks include:

- Predicting depth.
- Predicting surface normals.
- Reconstructing 3D shapes.
- Classifying 3D objects.
- Segmenting point clouds.
- Generating meshes.
- Synthesizing novel views.

---

## 72. 2.5D vs 3D

2.5D representations:

- Depth maps.
- Surface normals.
- View-dependent.
- Easier to predict from images.
- Do not fully represent complete 3D shape.

3D representations:

- Voxels.
- Point clouds.
- Meshes.
- Implicit functions.
- Represent actual 3D geometry.
- Require specialized architectures and losses.

---

## 73. Architectural Progression

The lecture shows a progression of methods:

1. Multi-view CNNs reuse 2D CNNs from multiple rendered views.
2. Depth and normal prediction use fully convolutional networks.
3. Voxels use 3D CNNs.
4. Point clouds use PointNet and permutation-invariant pooling.
5. Meshes use graph convolutions and iterative deformation.
6. Implicit functions use neural fields.
7. NeRF uses continuous radiance fields for novel view synthesis.
8. 3D Gaussian Splatting improves rendering speed using discrete Gaussians.

---

## 74. Main Challenges

The main challenges in 3D vision are:

- Choosing the right 3D representation.
- Handling memory growth in dense 3D grids.
- Processing unordered point sets.
- Designing losses for sets and surfaces.
- Recovering 3D from ambiguous 2D images.
- Handling occlusion and amodal completion.
- Rendering efficiently.
- Scaling methods to real-world scenes.

---

## 75. Practical Summary

For a computer vision project:

- Use depth maps if the output is per-pixel geometry.
- Use point clouds if the input comes from 3D sensors or LiDAR.
- Use PointNet-style models for point cloud classification or segmentation.
- Use voxels only if the resolution is manageable.
- Use meshes when explicit surfaces are needed.
- Use Chamfer distance when comparing unordered point sets.
- Use graph convolution when processing mesh connectivity.
- Use NeRF or Gaussian Splatting for novel view synthesis.
- Use datasets such as ShapeNet, Pix3D, and PartNet depending on the task.

# Lab 01: The VRAM Constraint and 4D Tensor Engineering

## 1. Lab Goal

This notebook studies the engineering constraints behind video models and 3D convolutional networks.

The main focus is not only model accuracy, but the practical problems that appear when working with video tensors:

- 4D video tensor layout.
- Temporal subsampling.
- Spatial resizing.
- GPU VRAM estimation.
- Data type mistakes.
- Autograd memory leaks.
- Temporal jitter as video augmentation.

The notebook uses a small curated UCF101-based video dataset.

Dataset slug:

    uvigo-video-understanding-lab-01

The dataset contains 33 short videos.

---

# 2. Environment Setup

## Purpose

The setup cell imports the required libraries, defines a helper function for loading video files, and checks that CUDA is available.

The lab is designed to run on a GPU because video tensors and 3D CNNs can quickly exceed CPU/GPU memory limits.

## Code

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import v2
    import cv2
    import numpy as np
    import os, glob, random, time
    import matplotlib.pyplot as plt

    def load_video(path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        return torch.from_numpy(np.stack(frames))  # [T, H, W, C], dtype: uint8

    if not torch.cuda.is_available():
        raise SystemError(
            'T4 GPU not detected. Go to Settings -> Accelerator -> GPU T4 x1 and restart.'
        )

    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1024**3

    print(f'GPU   : {gpu.name}')
    print(f'VRAM  : {vram_gb:.2f} GB')
    print(f'PyTorch: {torch.__version__}')

## Explanation

The `load_video` function reads a video file frame by frame using OpenCV.

OpenCV loads frames in BGR format, so the code converts each frame to RGB:

    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

The returned video tensor has shape:

    [T, H, W, C]

where:

- `T` = number of frames.
- `H` = frame height.
- `W` = frame width.
- `C` = color channels.

PyTorch 3D convolution expects the tensor layout to be different, so this raw decoder layout must be fixed later.

---

# 3. Task 1: The Silent Axis Crime

## Problem

A raw video tensor from the decoder has shape:

    [T, H, W, C]

But PyTorch `nn.Conv3d` expects batched video input in this format:

    [B, C, T, H, W]

For a single video clip without batch dimension, the correct format is:

    [C, T, H, W]

The task is to correctly permute axes before resizing.

## Corrected Code

    raw_video_tensor = torch.randint(0, 255, (4, 320, 240, 3), dtype=torch.uint8)

    pytorch_tensor = raw_video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]

    resize = v2.Resize((224, 224), antialias=True)
    final_video = resize(pytorch_tensor)

    print(f'Shape after permute+resize : {final_video.shape}')

    assert final_video.shape == (3, 4, 224, 224), (
        f'FAIL: expected (3, 4, 224, 224), got {final_video.shape}'
    )

    print('PASS')

## Explanation

The raw tensor shape is:

    [4, 320, 240, 3]

This means:

- 4 frames.
- Height 320.
- Width 240.
- 3 RGB channels.

The correct permutation is:

    permute(3, 0, 1, 2)

It transforms:

    [T, H, W, C]

into:

    [C, T, H, W]

So the tensor becomes:

    [3, 4, 320, 240]

After resizing, only the last two dimensions are changed:

    [3, 4, 224, 224]

Important detail:

`v2.Resize` resizes only the spatial dimensions at the end of the tensor. It does not verify whether the first dimensions actually mean channels and time. This is why wrong axis ordering can silently survive preprocessing.

---

# 4. Task 2: Building a Correct Video Dataset

## Problem

The custom dataset must load videos, sample frames, reorder axes, normalize pixels, and return tensors ready for a 3D CNN.

The correct output shape must be:

    [C, T, H, W] = [3, 16, 224, 224]

The final tensor must also be:

    torch.float32

with normalized pixel values in the range:

    [0, 1]

## Corrected Code

    DATASET_PATH = 'uvigo-video-ucf101-micro'

    class VideoDataset(Dataset):
        def __init__(self, data_path, num_frames=16, stride=2, jitter=True):
            class_names = sorted([
                d for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d))
            ])

            self.class_to_idx = {c: i for i, c in enumerate(class_names)}
            self.video_paths = []
            self.labels = []

            for cls in class_names:
                pattern = os.path.join(data_path, cls, '*.avi')
                for video_path in glob.glob(pattern):
                    self.video_paths.append(video_path)
                    self.labels.append(self.class_to_idx[cls])

            self.num_frames = num_frames
            self.stride = stride
            self.jitter = jitter
            self.transform = v2.Resize((224, 224), antialias=True)

        def __len__(self):
            return len(self.video_paths)

        def __getitem__(self, idx):
            video_tensor = load_video(self.video_paths[idx])  # [T, H, W, C], uint8

            total_frames = video_tensor.shape[0]
            required = self.num_frames * self.stride

            if self.jitter:
                # s_max = total_frames - num_frames * stride
                start = random.randint(
                    0,
                    total_frames - self.num_frames * self.stride
                )
                indices = list(range(start, start + required, self.stride))
            else:
                indices = list(range(0, required, self.stride))

            subsampled = video_tensor[indices]

            subsampled = subsampled.permute(3, 0, 1, 2)  # [C, T, H, W]

            final_tensor = subsampled.float() / 255.0

            return self.transform(final_tensor), torch.tensor(self.labels[idx])

    ds = VideoDataset(DATASET_PATH)

    for i in range(len(ds)):
        clip, label = ds[i]

        assert clip.shape == (3, 16, 224, 224), f'[{i}] shape: {clip.shape}'
        assert clip.dtype == torch.float32, f'[{i}] dtype: {clip.dtype}'
        assert clip.mean() > 0.01, f'[{i}] Darkness Bug active: mean={clip.mean():.4f}'

    print(f'All {len(ds)} clips passed.')

## Explanation

This dataset performs several important steps.

### Step 1: Load video

    video_tensor = load_video(self.video_paths[idx])

The loaded tensor has shape:

    [T, H, W, C]

and type:

    uint8

Pixel values are in:

    [0, 255]

### Step 2: Temporal subsampling

The dataset does not use every frame.

Instead, it samples:

    num_frames = 16

with temporal stride:

    stride = 2

This means it selects every second frame.

The number of frames needed from the original video is:

    required = num_frames * stride

For `num_frames=16` and `stride=2`:

    required = 32

The selected indices are:

    start, start + 2, start + 4, ..., start + 30

This returns exactly 16 frames.

### Step 3: Temporal jitter

If `jitter=True`, the starting frame is randomized.

This avoids always selecting the same clip from the video.

The maximum valid start frame is:

    s_max = total_frames - num_frames * stride

The formula ensures that the last selected frame stays inside the video.

### Step 4: Axis permutation

After indexing, `subsampled` still has shape:

    [T, H, W, C]

The correct permutation is:

    subsampled.permute(3, 0, 1, 2)

This gives:

    [C, T, H, W]

### Step 5: Normalize pixels

The code converts `uint8` pixels into normalized float values:

    final_tensor = subsampled.float() / 255.0

This produces:

    dtype = torch.float32

and values in:

    [0, 1]

This is the correct format for neural network input.

---

# 5. Task 2: Common Defects

## Defect 1: Ignoring Temporal Stride

Wrong version:

    indices = list(range(0, required, 1))

Problem:

This selects every frame instead of every `stride`-th frame.

For:

    num_frames = 16
    stride = 2
    required = 32

wrong code selects:

    32 frames

instead of:

    16 frames

Correct version:

    indices = list(range(0, required, self.stride))

Why the assertion catches it:

    assert clip.shape == (3, 16, 224, 224)

The wrong code gives temporal dimension 32, so the shape assertion fails.

---

## Defect 2: Wrong Axis Order

Wrong version:

    subsampled = subsampled.permute(0, 3, 1, 2)

Starting from:

    [T, H, W, C]

this produces:

    [T, C, H, W]

But PyTorch 3D CNNs need:

    [C, T, H, W]

Correct version:

    subsampled = subsampled.permute(3, 0, 1, 2)

Why the assertion catches it:

    assert clip.shape == (3, 16, 224, 224)

The wrong tensor has time and channel dimensions swapped.

---

## Defect 3: The Darkness Bug

Wrong version:

    final_tensor = (subsampled / 255).to(torch.uint8)

Problem:

After division by 255, pixel values are floats in:

    [0, 1]

Casting these values to `uint8` truncates them to integers.

Most values become:

    0

So the image becomes almost completely black.

Correct version:

    final_tensor = subsampled.float() / 255.0

Assertions that catch it:

    assert clip.dtype == torch.float32
    assert clip.mean() > 0.01

The dtype assertion catches that the tensor is not float.

The mean assertion catches that the tensor became almost all zeros.

---

# 6. Task 3: VRAM Arithmetic

## Problem

Before running a video model, we should calculate the input tensor memory.

A video tensor has shape:

    [B, C, T, H, W]

Given:

| Parameter         |   Value |
| ----------------- | ------: |
| Batch size B      |       8 |
| Channels C        |       3 |
| Frames T          |      16 |
| Height H          |     224 |
| Width W           |     224 |
| Data type         | float32 |
| Bytes per float32 |       4 |

## Correct Code

    B, C, T, H, W = 8, 3, 16, 224, 224
    BYTES_PER_FLOAT32 = 4

    VRAM_bytes = B * C * T * H * W * BYTES_PER_FLOAT32

    VRAM_MB = VRAM_bytes / 1024**2
    VRAM_GB = VRAM_bytes / 1024**3

    print(f'Baseline  : {VRAM_bytes:,} bytes  |  {VRAM_MB:.1f} MB  |  {VRAM_GB:.3f} GB')

    T_r = T // 2
    bytes_r1 = B * C * T_r * H * W * BYTES_PER_FLOAT32

    assert bytes_r1 == VRAM_bytes // 2, (
        f'R1 failed: {bytes_r1} != {VRAM_bytes // 2}'
    )

    print(f'After T-stride k=2 : {bytes_r1 / 1024**2:.1f} MB')

    bytes_r2 = B * C * T_r * (H // 2) * (W // 2) * BYTES_PER_FLOAT32

    assert bytes_r2 == bytes_r1 // 4, (
        f'R2 failed: {bytes_r2} != {bytes_r1 // 4}'
    )

    print(f'After 112x112      : {bytes_r2 / 1024**2:.1f} MB')

    bytes_r3 = bytes_r2 // 2

    assert bytes_r3 == bytes_r2 // 2, (
        f'R3 failed: {bytes_r3} != {bytes_r2 // 2}'
    )

    print(f'After float16      : {bytes_r3 / 1024**2:.1f} MB')

    print(f'Total reduction factor : {VRAM_bytes / bytes_r3:.1f}x')

    T4_VRAM_GB = 16.0
    training_vram_gb = 3 * VRAM_GB
    fits = training_vram_gb <= T4_VRAM_GB

    print(
        f'Training VRAM (3× input): {training_vram_gb:.3f} GB — '
        f'{"FITS" if fits else "DOES NOT FIT"} on T4 ({T4_VRAM_GB} GB)'
    )

    save_r1 = VRAM_bytes - bytes_r1
    save_r2 = bytes_r1 - bytes_r2
    save_r3 = bytes_r2 - bytes_r3

    savings = {
        'T-stride k=2': save_r1,
        '112×112 resize': save_r2,
        'float16 cast': save_r3,
    }

    best = max(savings, key=savings.get)

    for name, val in savings.items():
        print(f'  {name}: {val:,} bytes saved ({val / 1024**2:.1f} MB)')

    print(f'Largest absolute saving: {best}')

## Output Values

Baseline input tensor:

    77,070,336 bytes
    73.5 MB
    0.072 GB

After temporal stride `k=2`:

    36.8 MB

After spatial resize to `112 x 112`:

    9.2 MB

After float16:

    4.6 MB

Total reduction:

    16.0x

## Explanation

The baseline formula is:

    B * C * T * H * W * bytes_per_value

Substituting the values:

    8 * 3 * 16 * 224 * 224 * 4 = 77,070,336 bytes

This equals about:

    73.5 MB

The temporal dimension matters. If `T` is missing from the formula, the memory estimate is wrong by a factor of:

    16x

because the tensor contains 16 frames per clip.

---

## Reduction 1: Temporal Stride

Halving the number of frames halves memory.

Original:

    T = 16

After stride reduction:

    T = 8

Memory reduction:

    2x

---

## Reduction 2: Spatial Resize

Changing resolution from:

    224 x 224

to:

    112 x 112

halves both height and width.

Area reduction:

    2 * 2 = 4x

So memory becomes four times smaller for the same number of frames.

---

## Reduction 3: Float16

Float32 uses:

    4 bytes

Float16 uses:

    2 bytes

So casting to float16 halves memory.

---

# 7. Task 4: The Ghost Gradient Leak

## Problem

The model and optimizer are correct, but memory can grow if tensors connected to the autograd graph are stored in a Python list.

The problematic line is:

    history.append(output)

because `output` still has a `grad_fn` and keeps the computational graph alive.

## Model Code

    class Simple3DCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv3d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2)
            )

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 8 * 112 * 112, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.head(self.conv(x))

## Training Loop

    def run_training_loop(label='TEST', n_iterations=10, retain_output=False):
        torch.cuda.empty_cache()

        model = Simple3DCNN().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        history = []

        print(f'--- {label} ---')

        vram_before = torch.cuda.memory_allocated()
        print(f'VRAM before loop : {vram_before / 1024**2:.0f} MB')

        vram_after_first = None

        for i in range(n_iterations):
            data = torch.randn(8, 3, 16, 224, 224).cuda()
            target = torch.randint(0, 2, (8,)).cuda()

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if retain_output:
                history.append(output)  # Leaky version: keeps autograd graph alive

            if i == 0:
                vram_after_first = torch.cuda.memory_allocated()

        vram_after = torch.cuda.memory_allocated()

        print(f'VRAM after 1st iteration : {vram_after_first / 1024**2:.0f} MB')
        print(f'VRAM after {n_iterations} iterations : {vram_after / 1024**2:.2f} MB')
        print(f'Growth after 1st iter    : {(vram_after - vram_after_first) / 1024**2:.3f} MB')
        print(f'Tensors retained in history : {len(history)}')

        del model, optimizer, history
        torch.cuda.empty_cache()

        return vram_after_first, vram_after

    _, vram_leaky_end = run_training_loop(
        'LEAKY (sabotaged)',
        retain_output=True
    )

    vram_clean_first, vram_clean_end = run_training_loop(
        'CLEAN (fixed)',
        retain_output=False
    )

    assert vram_clean_end == vram_clean_first, 'CLEAN loop is still leaking memory'

    print('CLEAN LOOP PASSED')

## Explanation

The output tensor is connected to the computation graph.

When storing it directly:

    history.append(output)

Python keeps a reference to:

- the output tensor,
- its `grad_fn`,
- previous intermediate tensors needed for backward,
- graph metadata.

As a result, PyTorch cannot free the graph after backpropagation.

Correct options:

Option 1: do not store the output.

    pass

Option 2: store a detached tensor.

    history.append(output.detach())

Option 3: store CPU values only.

    history.append(output.detach().cpu())

Option 4: store scalar/log values.

    history.append(output.detach().cpu().numpy())

Important:

    torch.cuda.empty_cache()

cannot free tensors that are still referenced by Python objects.

It only releases unused cached memory blocks.

---

# 8. Task 5: Temporal Jitter

## Problem

If a video dataset always starts sampling from frame 0, the model sees the exact same 4D tensor every epoch.

For small datasets, this can cause memorization.

Temporal jitter fixes this by randomly choosing a valid start frame.

## Verification Code

    ds_jitter = VideoDataset(DATASET_PATH, jitter=True)

    clip1, _ = ds_jitter[0]
    clip2, _ = ds_jitter[0]

    ds_fixed = VideoDataset(DATASET_PATH, jitter=False)

    clip3, _ = ds_fixed[0]
    clip4, _ = ds_fixed[0]

    print(
        f"Jitter ON  — same clip called twice, tensors differ : "
        f"{not torch.equal(clip1, clip2)}"
    )

    print(
        f"Jitter OFF — same clip called twice, tensors equal  : "
        f"{torch.equal(clip3, clip4)}"
    )

    assert not torch.equal(clip1, clip2), (
        "FAIL: jitter=True returned identical tensors"
    )

    assert torch.equal(clip3, clip4), (
        "FAIL: jitter=False returned different tensors"
    )

    print("PASS")

## Explanation

With jitter enabled:

    ds_jitter[0]

can return a different temporal slice each time.

With jitter disabled:

    ds_fixed[0]

always starts from frame 0 and returns the same tensor.

Temporal jitter creates different training clips from the same video.

This helps the model learn the action rather than memorizing one fixed frame sequence.

---

# 9. Maximum Valid Start Index

## Formula

Let:

- `F` = total frames in the video.
- `N` = number of frames required in the output clip.
- `k` = temporal stride.
- `s` = start frame.

The sampled frames are:

    s, s + k, s + 2k, ..., s + (N - 1)k

The last index must be inside the video:

    s + (N - 1)k <= F - 1

So:

    s <= F - Nk

Therefore:

    s_max = F - N * k

In code:

    start = random.randint(0, total_frames - self.num_frames * self.stride)

Example:

    F = 150
    N = 16
    k = 2

Then:

    s_max = 150 - 16 * 2 = 118

Valid start positions:

    0, 1, 2, ..., 118

Number of possible start positions:

    119

---

# 10. Bug Audit Summary

| #   | Location                         | Principle Violated       | Minimal Fix                            | Why the Assertion Catches It      |
| --- | -------------------------------- | ------------------------ | -------------------------------------- | --------------------------------- |
| 1   | Task 1: wrong `permute`          | Axis semantics           | `permute(3, 0, 1, 2)`                  | Checks `[C, T, H, W]`             |
| 2   | Task 2: `range(..., 1)`          | Temporal subsampling     | `range(..., self.stride)`              | Checks `T = 16`                   |
| 3   | Task 2: wrong `permute`          | Axis semantics           | `permute(3, 0, 1, 2)`                  | Checks channel-first layout       |
| 4   | Task 2: cast to `uint8`          | Dtype propagation        | `subsampled.float() / 255.0`           | Checks `float32` and nonzero mean |
| 5   | Task 4: `history.append(output)` | Autograd graph retention | Remove append or use `output.detach()` | Checks VRAM stability             |

---

# 11. Activation Memory in Simple3DCNN

Input shape:

    [8, 3, 16, 224, 224]

The model:

    Conv3d(3 -> 16, kernel_size=3, padding=1)
    ReLU
    MaxPool3d(2)
    Flatten
    Linear(16 * 8 * 112 * 112 -> 128)
    ReLU
    Linear(128 -> 2)

Intermediate activations:

| Layer         | Output shape            |
| ------------- | ----------------------- |
| Conv3d        | `[8, 16, 16, 224, 224]` |
| ReLU          | `[8, 16, 16, 224, 224]` |
| MaxPool3d     | `[8, 16, 8, 112, 112]`  |
| Flatten       | `[8, 1605632]`          |
| Linear to 128 | `[8, 128]`              |
| Linear to 2   | `[8, 2]`                |

Activation memory calculation:

| Layer            |    Elements |       Bytes |
| ---------------- | ----------: | ----------: |
| Conv3d output    | 102,760,448 | 411,041,792 |
| ReLU output      | 102,760,448 | 411,041,792 |
| MaxPool3d output |  12,845,056 |  51,380,224 |
| Flatten          |  12,845,056 |  51,380,224 |
| Linear to 128    |       1,024 |       4,096 |
| Linear to 2      |          16 |          64 |

Total:

    924,848,192 bytes

In MB:

    882.0 MB

This shows that 3D CNNs can be activation-bound.

The model may not have huge weights, but the intermediate tensors are large because they scale with:

    B * C * T * H * W

---

# 12. The MLP Paradox

## Problem

A naive MLP receives a flattened video tensor:

    3 * 16 * 224 * 224

Number of input features:

    2,408,448

If the first layer is:

    nn.Linear(2,408,448, 256)

then parameter count is:

    2,408,448 * 256 + 256

Result:

    616,562,944 parameters

Weight memory in float32:

    616,562,944 * 4 bytes = 2,466,251,776 bytes

In MB:

    2352.0 MB

## Explanation

A naive MLP is weight-bound because it creates a separate weight for every input feature and every hidden unit.

A 3D CNN is usually activation-bound because convolutional weights are shared across space and time.

3D CNNs avoid the huge parameter count of MLPs by reusing the same kernels across the entire spatio-temporal volume.

---

# 13. Why Shape Checks Matter

Some video bugs are visually hard to notice.

Example:

Wrong stride code:

    range(0, required, 1)

For:

    num_frames = 16
    stride = 2
    required = 32

This selects 32 consecutive frames.

Visually, the clip still looks plausible because it is a smooth video segment.

But the tensor shape is wrong:

Before fix:

    [32, H, W, C]

After fix:

    [16, H, W, C]

After permutation:

Before fix:

    [C, 32, H, W]

After fix:

    [C, 16, H, W]

The assertion catches this:

    assert clip.shape == (3, 16, 224, 224)

Human visual inspection may not catch it because both versions look like valid videos.

---

# 14. Key Takeaways

## Tensor Layout

Raw video loaders often return:

    [T, H, W, C]

PyTorch `Conv3d` expects:

    [B, C, T, H, W]

For one clip, convert to:

    [C, T, H, W]

using:

    permute(3, 0, 1, 2)

---

## Temporal Subsampling

Use stride correctly:

    indices = list(range(start, start + required, stride))

where:

    required = num_frames * stride

This gives exactly `num_frames` sampled frames.

---

## Pixel Normalization

Correct normalization:

    tensor.float() / 255.0

Do not cast normalized values back to `uint8`.

---

## VRAM Formula

Input tensor memory:

    B * C * T * H * W * bytes_per_value

The temporal dimension must not be omitted.

---

## Memory Reductions

Main memory reduction strategies:

| Strategy          | Effect                 |
| ----------------- | ---------------------- |
| Reduce T          | Fewer frames           |
| Reduce H and W    | Fewer pixels per frame |
| Use float16       | Fewer bytes per value  |
| Reduce batch size | Fewer clips at once    |

Spatial resizing from `224 x 224` to `112 x 112` gives a 4x reduction in spatial area.

---

## Autograd Memory Leaks

Do not store tensors connected to the computation graph.

Leaky:

    history.append(output)

Safe:

    history.append(output.detach())

or:

    history.append(output.detach().cpu())

---

## Temporal Jitter

Temporal jitter prevents memorization by changing the start frame.

Correct maximum start formula:

    s_max = total_frames - num_frames * stride

Temporal jitter helps a 3D CNN learn action patterns instead of memorizing one fixed clip.

# Lab 02: The SlowFast Pipeline

## 1. Lab Goal

This notebook builds a minimal video classification pipeline based on the SlowFast architecture and ViViT-style tubelet tokenisation.

The lab focuses on three main engineering problems:

1. Dual-stream temporal sampling for SlowFast.
2. Building a minimal SlowFast network with a lateral connection.
3. Converting a video volume into transformer tokens using tubelets.

The pipeline is staged so that each part produces a verified tensor for the next part. If an earlier stage is broken, the later stages fail.

---

## 2. Architecture Parameters

The lab uses reduced parameters compared with the lecture version so that the code can run on a NVIDIA T4 GPU.

| Parameter | Lab value | Lecture value | Meaning                              |
| --------- | --------: | ------------: | ------------------------------------ |
| tau       |         4 |            16 | Slow pathway temporal stride         |
| alpha     |         4 |             8 | Fast/slow frame-rate ratio           |
| beta      |       1/8 |           1/8 | Fast/slow channel ratio              |
| T_slow    |         8 |             8 | Number of frames in the slow pathway |
| T_fast    |        32 |            64 | Number of frames in the fast pathway |

Important implementation detail:

In the code, `BETA = 8` is used as a denominator:

    fast_channels = slow_channels // BETA

So mathematically:

    beta = 1 / BETA = 1/8

---

## 3. Dataset

The lab uses a curated dataset extracted from UCF101.

Dataset name:

    uvigo-video-classification-lab-02

Dataset path in Kaggle:

    /kaggle/input/datasets/ivnrodrguezconde/uvigo-video-classification-lab-02/uvigo-video-ucf101-small

The dataset uses CSV split files such as:

    train.csv

Each CSV row contains:

- clip path
- class label

---

# 4. Environment Setup

## Code

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import v2
    import cv2
    import numpy as np
    import os, glob, random
    import matplotlib.pyplot as plt
    import pandas as pd

    # Video loading helper using OpenCV
    def load_video(path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f'No frames decoded from {path}')

        return torch.from_numpy(np.stack(frames))  # [T, H, W, C], uint8

    # Hardware check
    if not torch.cuda.is_available():
        raise SystemError(
            'T4 GPU not detected. Go to Settings -> Accelerator -> GPU T4 x1 and restart.'
        )

    DATASET_PATH = '/kaggle/input/datasets/ivnrodrguezconde/uvigo-video-classification-lab-02/uvigo-video-ucf101-small'

    # SlowFast hyperparameters
    TAU = 4
    ALPHA = 4
    BETA = 8

    T_SLOW = 8
    T_FAST = T_SLOW * ALPHA

    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1024**3

    print(f'GPU    : {gpu.name}')
    print(f'VRAM   : {vram_gb:.2f} GB')
    print(f'PyTorch: {torch.__version__}')
    print(f'T_SLOW : {T_SLOW}  T_FAST : {T_FAST}')

## Explanation

The `load_video` function reads a video file frame by frame using OpenCV.

OpenCV decodes frames in BGR format, so the code converts each frame to RGB:

    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

The returned tensor has shape:

    [T, H, W, C]

where:

- T = number of frames
- H = height
- W = width
- C = color channels

This is not yet the format required by `nn.Conv3d`.

PyTorch 3D convolution expects batched input in this format:

    [B, C, T, H, W]

For a single clip inside a dataset, the correct format is:

    [C, T, H, W]

---

# 5. Stage 1: Dual-Stream Temporal Sampling

## Goal

SlowFast requires two clips from the same video:

1. Slow pathway clip.
2. Fast pathway clip.

The slow pathway samples fewer frames with a larger stride.

The fast pathway samples more frames with a smaller stride.

Both clips must come from the same temporal window so that they describe the same action segment.

---

## Sampling Rules

Slow pathway:

    T_slow frames
    stride = tau

Fast pathway:

    T_fast = alpha * T_slow
    stride = tau / alpha

In the lab:

    T_slow = 8
    tau = 4
    alpha = 4
    T_fast = 8 * 4 = 32
    fast_stride = tau / alpha = 4 / 4 = 1

So:

Slow pathway samples 8 frames every 4 frames.

Fast pathway samples 32 frames every 1 frame.

Both cover the same temporal span:

    slow_window = T_slow * tau = 8 * 4 = 32
    fast_window = T_fast * fast_stride = 32 * 1 = 32

---

## Correct Code

    class SlowFastDataset(Dataset):
        def __init__(
            self,
            data_path,
            split='train',
            t_slow=T_SLOW,
            tau=TAU,
            alpha=ALPHA,
            jitter=True
        ):
            df = pd.read_csv(os.path.join(data_path, f'{split}.csv'))

            class_names = sorted(df['label'].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}

            self.video_paths = [
                os.path.join(data_path, row['clip_path'].lstrip('/'))
                for _, row in df.iterrows()
            ]

            self.labels = [
                self.class_to_idx[row['label']]
                for _, row in df.iterrows()
            ]

            self.t_slow = t_slow
            self.t_fast = t_slow * alpha
            self.tau = tau
            self.alpha = alpha
            self.jitter = jitter

            self.transform = v2.Resize((224, 224), antialias=True)

        def __len__(self):
            return len(self.video_paths)

        def __getitem__(self, idx):
            frames = load_video(self.video_paths[idx])  # [T, H, W, C], uint8
            total = frames.shape[0]

            fast_stride = self.tau // self.alpha

            slow_window = self.t_slow * self.tau
            fast_window = self.t_fast * fast_stride
            window_size = max(slow_window, fast_window)

            s_max = max(0, total - window_size)

            # One shared temporal start for both pathways
            start = random.randint(0, s_max) if self.jitter else 0

            slow_indices = list(
                range(
                    start,
                    start + self.t_slow * self.tau,
                    self.tau
                )
            )

            fast_indices = list(
                range(
                    start,
                    start + self.t_fast * fast_stride,
                    fast_stride
                )
            )

            slow_clip = frames[slow_indices]  # [T_slow, H, W, C]
            fast_clip = frames[fast_indices]  # [T_fast, H, W, C]

            def prepare(clip):
                # [T, H, W, C] -> [T, C, H, W]
                clip = clip.permute(0, 3, 1, 2).float() / 255.0

                # Resize spatial dimensions H and W
                clip = self.transform(clip)

                # [T, C, H, W] -> [C, T, H, W]
                clip = clip.permute(1, 0, 2, 3)

                return clip

            return (
                prepare(slow_clip),
                prepare(fast_clip),
                torch.tensor(self.labels[idx])
            )

    # Verification
    ds = SlowFastDataset(DATASET_PATH)

    for i in range(len(ds)):
        slow, fast, label = ds[i]

        assert slow.shape == (3, T_SLOW, 224, 224), \
            f'[{i}] slow shape: {slow.shape}  expected (3, {T_SLOW}, 224, 224)'

        assert fast.shape == (3, T_FAST, 224, 224), \
            f'[{i}] fast shape: {fast.shape}  expected (3, {T_FAST}, 224, 224)'

        assert slow.dtype == torch.float32, f'[{i}] slow dtype: {slow.dtype}'
        assert fast.dtype == torch.float32, f'[{i}] fast dtype: {fast.dtype}'

    print(f'All {len(ds)} clips passed.')
    print(f'slow clip : {slow.shape}  |  fast clip : {fast.shape}')

---

## Explanation

The dataset returns three values:

    slow_clip, fast_clip, label

The slow clip has shape:

    [3, 8, 224, 224]

The fast clip has shape:

    [3, 32, 224, 224]

Both are float tensors with pixel values normalized to:

    [0, 1]

---

## Why Both Pathways Need the Same Start Frame

SlowFast is not two independent classifiers.

The fast pathway is fused into the slow pathway using a lateral connection.

Therefore, the slow and fast features must describe the same action segment.

Wrong idea:

    slow_start = random.randint(...)
    fast_start = random.randint(...)

Problem:

The slow pathway could see one moment of the video, while the fast pathway sees a different moment.

Correct idea:

    start = random.randint(0, s_max)

Then both pathways use this same `start`.

---

## Correct Temporal Indices

For the slow pathway:

    slow_indices = range(start, start + T_slow * tau, tau)

For the fast pathway:

    fast_indices = range(start, start + T_fast * fast_stride, fast_stride)

With lab values:

    T_slow = 8
    tau = 4
    T_fast = 32
    fast_stride = 1

Example if `start = 0`:

Slow indices:

    0, 4, 8, 12, 16, 20, 24, 28

Fast indices:

    0, 1, 2, 3, ..., 31

Both pathways cover the same 32-frame temporal window.

---

## Correct Axis Order

Raw video from OpenCV:

    [T, H, W, C]

After first permutation:

    [T, C, H, W]

This is useful for resizing because torchvision resize expects spatial dimensions at the end.

After resize:

    [T, C, 224, 224]

Final permutation:

    [C, T, 224, 224]

This final format is correct for `nn.Conv3d` after batching:

    [B, C, T, H, W]

---

## Common Bugs in Stage 1

| Bug                                             | Why it is wrong                         | Correct fix                     |
| ----------------------------------------------- | --------------------------------------- | ------------------------------- |
| Different random starts for slow and fast clips | Breaks temporal correspondence          | Use one shared `start`          |
| Slow pathway uses `fast_stride`                 | Slow pathway becomes too dense          | Use `self.tau` for slow indices |
| Output stays `[T, C, H, W]`                     | Wrong format for `nn.Conv3d`            | Convert to `[C, T, H, W]`       |
| Missing `.float() / 255.0`                      | Pixel values remain uint8 in `[0, 255]` | Normalize to float32            |

---

# 6. Stage 2: Minimal SlowFast Network

## Goal

This stage builds a minimal SlowFast network.

It has:

1. Slow pathway.
2. Fast pathway.
3. Lateral connection.
4. Final classifier.

The lateral connection fuses fast-pathway motion features into the slow pathway.

---

## Correct Code

    class MinimalSlowFast(nn.Module):
        def __init__(self, num_classes=2, beta=BETA):
            super().__init__()

            slow_C = 64
            fast_C = slow_C // beta   # 64 / 8 = 8

            # Slow pathway: input [B, 3, T_slow, H, W]
            self.slow_conv = nn.Sequential(
                nn.Conv3d(
                    3,
                    slow_C,
                    kernel_size=(1, 7, 7),
                    stride=(1, 2, 2),
                    padding=(0, 3, 3)
                ),
                nn.ReLU(),
                nn.MaxPool3d(
                    kernel_size=(1, 3, 3),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1)
                )
            )

            # Fast pathway: input [B, 3, T_fast, H, W]
            self.fast_conv = nn.Sequential(
                nn.Conv3d(
                    3,
                    fast_C,
                    kernel_size=(3, 7, 7),
                    stride=(1, 2, 2),
                    padding=(1, 3, 3)
                ),
                nn.ReLU(),
                nn.MaxPool3d(
                    kernel_size=(1, 3, 3),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1)
                )
            )

            # Lateral connection:
            # 1. Reduces T_fast to T_slow.
            # 2. Projects fast_C channels to slow_C channels.
            self.lateral = nn.Conv3d(
                in_channels=fast_C,
                out_channels=slow_C,
                kernel_size=(5, 1, 1),
                stride=(ALPHA, 1, 1),
                padding=(2, 0, 0)
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(2 * slow_C, num_classes)
            )

        def forward(self, slow, fast):
            slow_feat = self.slow_conv(slow)      # [B, slow_C, T_slow, H', W']
            fast_feat = self.fast_conv(fast)      # [B, fast_C, T_fast, H', W']

            fused_fast = self.lateral(fast_feat)  # [B, slow_C, T_slow, H', W']

            # Concatenate along channel dimension
            fused = torch.cat([slow_feat, fused_fast], dim=1)

            return self.classifier(fused)

    # Verification
    loader = DataLoader(
        SlowFastDataset(DATASET_PATH),
        batch_size=4,
        shuffle=False
    )

    slow_batch, fast_batch, labels = next(iter(loader))

    slow_batch = slow_batch.cuda()
    fast_batch = fast_batch.cuda()
    labels = labels.cuda()

    num_classes = len(SlowFastDataset(DATASET_PATH).class_to_idx)

    model = MinimalSlowFast(num_classes=num_classes).cuda()

    with torch.no_grad():
        output = model(slow_batch, fast_batch)

    assert output.shape == (4, num_classes), \
        f'FAIL: expected ({4}, {num_classes}), got {output.shape}'

    print(f'PASS  |  output shape : {output.shape}')

    # Intermediate shape verification
    slow_feat = model.slow_conv(slow_batch)
    fast_feat = model.fast_conv(fast_batch)
    fused_fast = model.lateral(fast_feat)

    assert slow_feat.shape[2] == T_SLOW, \
        f'slow_feat temporal dim: {slow_feat.shape[2]}  expected {T_SLOW}'

    assert fused_fast.shape[2] == T_SLOW, \
        f'fused_fast temporal dim: {fused_fast.shape[2]}  expected {T_SLOW}'

    print(f'slow_feat  : {slow_feat.shape}')
    print(f'fast_feat  : {fast_feat.shape}')
    print(f'fused_fast : {fused_fast.shape}')

---

## Expected Output Shapes

For a batch size of 4:

Slow input:

    [4, 3, 8, 224, 224]

Fast input:

    [4, 3, 32, 224, 224]

After the slow pathway:

    [4, 64, 8, 56, 56]

After the fast pathway:

    [4, 8, 32, 56, 56]

After lateral connection:

    [4, 64, 8, 56, 56]

After concatenation:

    [4, 128, 8, 56, 56]

After adaptive average pooling:

    [4, 128, 1, 1, 1]

After flattening:

    [4, 128]

Final classifier output:

    [4, num_classes]

---

## Lateral Connection

The lateral connection is:

    nn.Conv3d(
        in_channels=8,
        out_channels=64,
        kernel_size=(5, 1, 1),
        stride=(4, 1, 1),
        padding=(2, 0, 0)
    )

Its job is to convert:

    [B, 8, 32, 56, 56]

into:

    [B, 64, 8, 56, 56]

It does two things at once:

1. Reduces the temporal dimension from 32 to 8.
2. Projects channels from 8 to 64.

---

## Temporal Output Size Formula

For a 3D convolution along the temporal dimension:

    T_out = floor((T_in + 2p - k) / s) + 1

For the lateral connection:

    T_in = 32
    p = 2
    k = 5
    s = ALPHA = 4

So:

    T_out = floor((32 + 2*2 - 5) / 4) + 1
    T_out = floor((32 + 4 - 5) / 4) + 1
    T_out = floor(31 / 4) + 1
    T_out = 7 + 1
    T_out = 8

This matches `T_slow`.

---

## Lateral Connection Parameter Count

Given:

    slow_C = 64
    fast_C = 8
    kernel_size = 5 x 1 x 1

Weight parameters:

    out_channels * in_channels * k_t * k_h * k_w

So:

    64 * 8 * 5 * 1 * 1 = 2560

Bias parameters:

    64

Total:

    2560 + 64 = 2624

---

## Why the Classifier Uses `2 * slow_C`

Before fusion:

    slow_feat has 64 channels
    fused_fast has 64 channels

Concatenation is done along the channel dimension:

    torch.cat([slow_feat, fused_fast], dim=1)

Therefore:

    fused channels = 64 + 64 = 128

The classifier must use:

    nn.Linear(2 * slow_C, num_classes)

not:

    nn.Linear(slow_C, num_classes)

---

## Common Bugs in Stage 2

| Bug                                     | Result                                                                       | Correct fix                                 |
| --------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------- |
| Lateral stride is `(1, 1, 1)`           | Fast temporal dimension remains 32, cannot concatenate with slow dimension 8 | Use `stride=(ALPHA, 1, 1)`                  |
| Classifier uses `slow_C` input features | Linear layer receives 128 features but expects 64                            | Use `nn.Linear(2 * slow_C, num_classes)`    |
| Concatenation along wrong dimension     | Breaks feature semantics                                                     | Concatenate along channel dimension `dim=1` |

---

# 7. Stage 3: Tubelet Tokenisation

## Goal

ViViT-style models convert a video volume into a sequence of spatiotemporal tokens.

A tubelet is a small 3D patch covering:

- several frames
- a spatial patch in height
- a spatial patch in width

The tubelet size is:

    t_patch x patch_size x patch_size

The Conv3D kernel and stride must match the tubelet dimensions to produce non-overlapping tubelets.

---

## Token Count Formula

The number of tokens is:

    N = (T / t_patch) * (H / patch_size) * (W / patch_size)

For:

    T = 16
    H = 224
    W = 224
    t_patch = 2
    patch_size = 16

we get:

    N = (16 / 2) * (224 / 16) * (224 / 16)
    N = 8 * 14 * 14
    N = 1568

---

## Correct Code

    class TubeletEmbedder(nn.Module):
        def __init__(
            self,
            t_patch=2,
            patch_size=16,
            embed_dim=768,
            num_frames=16,
            img_size=224
        ):
            super().__init__()

            self.t_patch = t_patch
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.num_frames = num_frames
            self.img_size = img_size

            # Kernel and stride match to create non-overlapping tubelets
            self.proj = nn.Conv3d(
                in_channels=3,
                out_channels=embed_dim,
                kernel_size=(t_patch, patch_size, patch_size),
                stride=(t_patch, patch_size, patch_size)
            )

        def token_count(self):
            return (
                (self.num_frames // self.t_patch)
                * (self.img_size // self.patch_size)
                * (self.img_size // self.patch_size)
            )

        def forward(self, x):
            # x: [B, C, T, H, W]
            tokens = self.proj(x)          # [B, D, T', H', W']
            tokens = tokens.flatten(2)     # [B, D, N]
            tokens = tokens.transpose(1, 2) # [B, N, D]
            return tokens

    # Verification
    T_VIT = 16
    P = 16
    TP = 2
    D = 768
    B = 2

    embedder = TubeletEmbedder(
        t_patch=TP,
        patch_size=P,
        embed_dim=D,
        num_frames=T_VIT,
        img_size=224
    )

    video = torch.randn(B, 3, T_VIT, 224, 224)
    tokens = embedder(video)

    expected_N = (T_VIT // TP) * (224 // P) * (224 // P)

    assert embedder.token_count() == expected_N, \
        f'token_count(): {embedder.token_count()}  expected {expected_N}'

    assert tokens.shape == (B, expected_N, D), \
        f'FAIL: expected ({B}, {expected_N}, {D}), got {tokens.shape}'

    print(f'PASS')
    print(f'Tokens per video : {expected_N}')
    print(f'Token tensor     : {tokens.shape}  [B, N, D]')

---

## Explanation

Input video shape:

    [B, C, T, H, W]

Example:

    [2, 3, 16, 224, 224]

The 3D convolution creates tubelet embeddings.

After Conv3D:

    [B, D, T', H', W']

where:

    D = embedding dimension
    T' = T / t_patch
    H' = H / patch_size
    W' = W / patch_size

For the lab values:

    T' = 16 / 2 = 8
    H' = 224 / 16 = 14
    W' = 224 / 16 = 14

So the Conv3D output is:

    [2, 768, 8, 14, 14]

A transformer cannot directly consume this 5D tensor.

A transformer expects:

    [B, N, D]

where:

- B = batch size
- N = sequence length / number of tokens
- D = embedding dimension

So the code flattens the temporal and spatial dimensions:

    tokens = tokens.flatten(2)

This converts:

    [B, D, T', H', W']

into:

    [B, D, N]

Then it transposes:

    tokens = tokens.transpose(1, 2)

This gives:

    [B, N, D]

Final shape:

    [2, 1568, 768]

---

## Why Stride Must Match Kernel Size

For non-overlapping tubelets:

    kernel_size = (t_patch, patch_size, patch_size)
    stride = (t_patch, patch_size, patch_size)

If temporal stride is smaller than `t_patch`, tubelets overlap in time.

Example of wrong stride:

    stride = (1, patch_size, patch_size)

With:

    T = 16
    t_patch = 2
    temporal stride = 1

Temporal output size:

    T_out = floor((16 - 2) / 1) + 1
    T_out = 15

Spatial output:

    H_out = 14
    W_out = 14

Wrong token count:

    N = 15 * 14 * 14 = 2940

Correct token count:

    N = 8 * 14 * 14 = 1568

So the wrong stride creates too many tokens and increases attention cost.

---

# 8. Integration Test

## Goal

The integration test verifies that:

1. The dataset produces valid slow and fast clips.
2. The MinimalSlowFast model accepts those clips.
3. The model produces class logits.
4. Cross-entropy loss is finite and positive.
5. Backpropagation works.

---

## Code

    torch.cuda.empty_cache()

    loader = DataLoader(
        SlowFastDataset(DATASET_PATH, jitter=True),
        batch_size=4,
        shuffle=True
    )

    slow_b, fast_b, labels_b = next(iter(loader))

    slow_b = slow_b.cuda()
    fast_b = fast_b.cuda()
    labels_b = labels_b.cuda()

    num_classes = len(SlowFastDataset(DATASET_PATH).class_to_idx)

    model = MinimalSlowFast(num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    output = model(slow_b, fast_b)
    loss = criterion(output, labels_b)

    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f'Loss is not finite: {loss.item()}'
    assert loss.item() > 0, f'Loss is zero or negative: {loss.item()}'

    print(f'PASS  |  loss = {loss.item():.4f}')
    print(f'VRAM used : {torch.cuda.memory_allocated()/1024**2:.0f} MB')

## Explanation

This cell checks the full pipeline:

    dataset -> slow/fast clips -> SlowFast model -> loss -> backward pass

If any earlier stage is wrong, this cell will fail.

Examples:

- Wrong dataset axis order causes `Conv3d` to receive invalid dimensions.
- Wrong lateral stride causes temporal mismatch during concatenation.
- Wrong classifier input size causes a linear layer shape error.
- Wrong label or output shape causes cross-entropy to fail.

---

# 9. Important Engineering Notes

## 9.1 Temporal Correspondence Invariant

Slow and fast pathways must describe the same temporal window.

Slow pathway:

- captures slower semantic appearance information
- lower frame rate
- higher channel capacity

Fast pathway:

- captures motion information
- higher frame rate
- lower channel capacity

If the pathways observe different video segments, the lateral connection fuses unrelated information.

Example failure:

- Slow pathway sees a person preparing to jump.
- Fast pathway sees the person already landing.

The lateral connection then learns noisy associations instead of useful motion-semantic relationships.

---

## 9.2 Valid Start Position for Jitter

The maximum valid start position is:

    s_max = total_frames - window_size

For SlowFast:

    window_size = max(T_slow * tau, T_fast * fast_stride)

Since:

    T_fast = T_slow * alpha
    fast_stride = tau / alpha

then:

    T_fast * fast_stride = T_slow * alpha * tau / alpha
    T_fast * fast_stride = T_slow * tau

So both pathways cover the same temporal window:

    window_size = T_slow * tau

For a 10-second video at 30 FPS:

    total_frames = 300

Lecture parameters:

    tau = 16
    T_slow = 8
    window_size = 8 * 16 = 128
    s_max = 300 - 128 = 172
    valid start positions = 173

Lab parameters:

    tau = 4
    T_slow = 8
    window_size = 8 * 4 = 32
    s_max = 300 - 32 = 268
    valid start positions = 269

The lab configuration allows more jitter positions because it uses a shorter temporal window.

---

## 9.3 Lateral Connection Budget

For one lateral connection without bias:

    params = C_out * C_in * k_t * k_h * k_w

In SlowFast:

    C_out = C_slow
    C_in = C_fast = C_slow / 8
    kernel = 5 x 1 x 1

So:

    params = C_slow * (C_slow / 8) * 5
    params = (5/8) * C_slow^2

For four residual stages:

| Stage | C_slow | C_fast |             Parameters |
| ----- | -----: | -----: | ---------------------: |
| 1     |     64 |      8 |     64 _ 8 _ 5 = 2,560 |
| 2     |    128 |     16 |  128 _ 16 _ 5 = 10,240 |
| 3     |    256 |     32 |  256 _ 32 _ 5 = 40,960 |
| 4     |    512 |     64 | 512 _ 64 _ 5 = 163,840 |

Total:

    2,560 + 10,240 + 40,960 + 163,840 = 217,600

Compared with a 23M-parameter slow ResNet-50 backbone:

    217,600 / (23,000,000 + 217,600) ≈ 0.00937

As a percentage:

    0.937%

So lateral connections add less than 1% of the backbone size.

---

## 9.4 Fast Pathway Activation Ratio

Ignoring batch size and spatial dimensions:

Slow pathway activation memory is proportional to:

    C_slow * T_slow

Fast pathway activation memory is proportional to:

    C_fast * T_fast

Using:

    C_fast = beta * C_slow
    T_fast = alpha * T_slow

Fast-to-slow ratio:

    (C_fast * T_fast) / (C_slow * T_slow)
    = (beta * C_slow * alpha * T_slow) / (C_slow * T_slow)
    = alpha * beta

For:

    alpha = 4
    beta = 1/8

Ratio:

    alpha * beta = 4 * 1/8 = 1/2

So the fast pathway uses about half as much activation memory as the slow pathway.

Fraction of total activation memory:

    fast / (slow + fast)
    = 0.5 / 1.5
    = 1/3

So the fast pathway consumes about 33.3% of the combined activation memory in this lab configuration.

---

# 10. Attention Complexity Notes

## 10.1 Joint Space-Time Attention

For video transformers:

    P = number of spatial patches per frame
    T = number of frames
    N = T * P

Joint attention complexity:

    O(N^2) = O((T * P)^2)

For:

    T = 16
    H = W = 224
    patch_size = 16

Spatial patches:

    P = (224 / 16)^2
    P = 14^2
    P = 196

Total tokens:

    N = T * P
    N = 16 * 196
    N = 3136

Attention matrix values:

    N^2 = 3136^2 = 9,834,496

Float32 memory:

    9,834,496 * 4 = 39,337,984 bytes

In GB:

    about 0.039 GB

This is only one attention matrix. Real training needs more memory for:

- Q, K, V projections
- activations
- gradients
- model parameters
- optimizer state
- multiple layers
- multiple heads

---

## 10.2 Factorised Attention: TimeSformer

TimeSformer factorises attention into:

1. Spatial attention.
2. Temporal attention.

Complexity:

    T * P^2 + P * T^2

For:

    T = 16
    P = 196

Spatial part:

    T * P^2 = 16 * 196^2
    196^2 = 38,416
    16 * 38,416 = 614,656

Temporal part:

    P * T^2 = 196 * 16^2
    16^2 = 256
    196 * 256 = 50,176

Total:

    614,656 + 50,176 = 664,832

Reduction factor compared with joint attention:

    9,834,496 / 664,832 ≈ 14.8

So factorised attention is about 14.8 times cheaper for this configuration.

---

# 11. Summary of Correct Tensor Shapes

## Dataset Output

Single sample:

| Tensor    | Shape               |
| --------- | ------------------- |
| slow clip | `[3, 8, 224, 224]`  |
| fast clip | `[3, 32, 224, 224]` |
| label     | scalar              |

Batch of 4:

| Tensor     | Shape                  |
| ---------- | ---------------------- |
| slow batch | `[4, 3, 8, 224, 224]`  |
| fast batch | `[4, 3, 32, 224, 224]` |
| labels     | `[4]`                  |

---

## MinimalSlowFast Intermediate Shapes

| Stage                 | Shape                  |
| --------------------- | ---------------------- |
| slow input            | `[4, 3, 8, 224, 224]`  |
| fast input            | `[4, 3, 32, 224, 224]` |
| slow features         | `[4, 64, 8, 56, 56]`   |
| fast features         | `[4, 8, 32, 56, 56]`   |
| lateral fast features | `[4, 64, 8, 56, 56]`   |
| fused features        | `[4, 128, 8, 56, 56]`  |
| classifier output     | `[4, num_classes]`     |

---

## Tubelet Tokenisation Shapes

Input:

    [2, 3, 16, 224, 224]

After Conv3D:

    [2, 768, 8, 14, 14]

After flattening:

    [2, 768, 1568]

After transpose:

    [2, 1568, 768]

Final transformer-ready format:

    [B, N, D]

---

# 12. Key Takeaways

## SlowFast Sampling

The slow pathway and fast pathway must sample from the same temporal window.

Correct logic:

    start = one shared random start

Slow pathway:

    stride = tau

Fast pathway:

    stride = tau / alpha

---

## SlowFast Architecture

The slow pathway has high channel capacity and low temporal resolution.

The fast pathway has low channel capacity and high temporal resolution.

The lateral connection fuses fast motion features into slow semantic features.

---

## Lateral Connection

The lateral connection must align temporal dimensions before concatenation.

Correct temporal stride:

    stride = ALPHA

Correct classifier input size after fusion:

    2 * slow_C

---

## Tubelet Tokenisation

A tubelet embedder uses Conv3D to create video tokens.

For non-overlapping tubelets:

    stride = kernel_size

The transformer expects:

    [B, N, D]

not:

    [B, D, T, H, W]

---

## Attention Cost

Joint space-time attention grows quadratically with all video tokens:

    (T * P)^2

Factorised attention reduces cost:

    T * P^2 + P * T^2

This is why architectures such as TimeSformer separate spatial and temporal attention.

---

## Practical Debugging Rules

Always verify:

- slow and fast clips come from the same temporal window
- slow clip shape is `[3, T_slow, H, W]`
- fast clip shape is `[3, T_fast, H, W]`
- model input batch shape is `[B, C, T, H, W]`
- lateral output temporal dimension equals `T_slow`
- fused channel count equals `2 * slow_C`
- transformer token shape is `[B, N, D]`

# Lab 03: Object Tracking — From Correlation Filters to Deep Matchers

## 1. Lab Goal

This notebook builds a single-object tracking evaluation pipeline based on Lesson 03.

The lab focuses on:

- loading tracking sequences correctly;
- running several trackers on the same sequences;
- computing standard tracking metrics;
- plotting success curves;
- finding tracker failure events;
- visualising predicted and ground-truth boxes;
- understanding why trackers fail.

The evaluation follows the OPE protocol.

OPE means One-Pass Evaluation.

In OPE:

1. The tracker is initialized on the first frame.
2. The tracker runs through the sequence once.
3. There is no manual reset after failure.
4. Predictions are compared with ground-truth boxes frame by frame.

---

# 2. Dataset and Trackers

## Dataset

The notebook uses a small LaSOT-based dataset.

Dataset path:

    /kaggle/input/datasets/ivnrodrguezconde/uvigo-object-tracking-lab-03/uvigo-video-lasot-small

Model directory:

    /kaggle/input/datasets/ivnrodrguezconde/uvigo-object-tracking-lab-03/uvigo-video-lasot-small/models

Sequences:

    car-17
    car-20
    person-7
    person-17
    bicycle-18
    bicycle-9

Ground-truth boxes are stored in:

    groundtruth.txt

Each ground-truth box has format:

    (x, y, w, h)

where:

- x = top-left x coordinate;
- y = top-left y coordinate;
- w = bounding box width;
- h = bounding box height.

---

## Trackers

The notebook compares three trackers:

| Tracker   | Type                       | Main idea                                                     |
| --------- | -------------------------- | ------------------------------------------------------------- |
| KCF       | Correlation-filter tracker | Uses kernelized correlation filters and a fixed search window |
| CSRT      | Correlation-filter tracker | Uses channel and spatial reliability maps                     |
| DaSiamRPN | Deep Siamese tracker       | Uses template matching and region proposal network            |

---

# 3. Environment Setup

## Code

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os, glob
    from typing import List, Tuple

    DATASET_PATH = '/kaggle/input/datasets/ivnrodrguezconde/uvigo-object-tracking-lab-03/uvigo-video-lasot-small'

    MODEL_DIR = '/kaggle/input/datasets/ivnrodrguezconde/uvigo-object-tracking-lab-03/uvigo-video-lasot-small/models'

    SEQUENCES = [
        'car-17',
        'car-20',
        'person-7',
        'person-17',
        'bicycle-18',
        'bicycle-9'
    ]

    def make_dasiamrpn():
        params = cv2.TrackerDaSiamRPN_Params()
        params.model       = f'{MODEL_DIR}/dasiamrpn_model.onnx'
        params.kernel_r1   = f'{MODEL_DIR}/dasiamrpn_kernel_r1.onnx'
        params.kernel_cls1 = f'{MODEL_DIR}/dasiamrpn_kernel_cls1.onnx'
        return cv2.TrackerDaSiamRPN_create(params)

    TRACKERS = {
        'KCF':       cv2.TrackerKCF_create,
        'CSRT':      cv2.TrackerCSRT_create,
        'DaSiamRPN': make_dasiamrpn,
    }

    print('OpenCV version:', cv2.__version__)
    print('Sequences      :', SEQUENCES)
    print('Trackers       :', list(TRACKERS.keys()))

## Output

    OpenCV version: 4.13.0
    Sequences      : ['car-17', 'car-20', 'person-7', 'person-17', 'bicycle-18', 'bicycle-9']
    Trackers       : ['KCF', 'CSRT', 'DaSiamRPN']

## Explanation

The code imports OpenCV, NumPy, Matplotlib, and helper modules.

OpenCV is used for:

- reading video frames;
- creating trackers;
- updating trackers frame by frame.

Matplotlib is used for:

- drawing bounding boxes;
- plotting success curves;
- plotting IoU over time.

DaSiamRPN requires three ONNX files:

    dasiamrpn_model.onnx
    dasiamrpn_kernel_r1.onnx
    dasiamrpn_kernel_cls1.onnx

These are passed through `cv2.TrackerDaSiamRPN_Params()`.

---

# 4. Stage 1: Evaluation Framework

## Goal

Stage 1 builds the evaluation functions.

The original framework contained three silent defects:

| Defect | Location                | Problem                                           |
| ------ | ----------------------- | ------------------------------------------------- |
| A      | `load_sequence`         | Frames may be loaded in wrong temporal order      |
| B      | `compute_iou`           | IoU union formula may use the wrong denominator   |
| C      | `compute_success_curve` | Success thresholds may start from the wrong value |

These defects do not necessarily crash the notebook, but they produce wrong metrics.

---

# 5. Loading a Tracking Sequence

## Code

    def load_sequence(dataset_path: str, sequence: str):
        frame_dir = os.path.join(dataset_path, sequence, 'img')
        gt_path   = os.path.join(dataset_path, sequence, 'groundtruth.txt')

        # Defect A fix:
        # glob does not guarantee temporal order.
        # Sorting keeps frames aligned with ground-truth annotations.
        frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

        with open(gt_path) as f:
            gt_boxes = [
                tuple(map(float, line.strip().split(',')))
                for line in f
            ]  # format: (x, y, w, h)

        return frame_files, gt_boxes

## Explanation

The function loads:

1. frame file paths from the `img` folder;
2. ground-truth bounding boxes from `groundtruth.txt`.

Important fix:

    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

`glob.glob()` does not guarantee temporal order.

If frames are not sorted, the tracker may process:

    frame 10 before frame 2

This breaks the tracking sequence.

Tracking depends on temporal order because each prediction depends on the previous frame.

Correct order is required so that:

    frame_files[i]

matches:

    gt_boxes[i]

---

# 6. IoU Metric

## Code

    def compute_iou(boxA: tuple, boxB: tuple) -> float:
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB

        ix = max(ax, bx)
        iy = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)

        inter = max(0.0, ix2 - ix) * max(0.0, iy2 - iy)

        # Defect B fix:
        # union must subtract the intersection once.
        union = aw * ah + bw * bh - inter

        return inter / union if union > 0 else 0.0

## Explanation

IoU means Intersection over Union.

Formula:

    IoU = area_of_intersection / area_of_union

For two boxes A and B:

    |A union B| = |A| + |B| - |A intersection B|

The intersection is subtracted once because it is counted in both box areas.

Wrong formula:

    union = area_A + area_B

Correct formula:

    union = area_A + area_B - intersection

Without subtracting the intersection, the union is too large and IoU becomes incorrect.

---

## Manual IoU Example

Given:

    boxA = (10.0, 10.0, 50.0, 50.0)
    boxB = (35.0, 35.0, 50.0, 50.0)

Box A spans:

    x: 10 to 60
    y: 10 to 60

Box B spans:

    x: 35 to 85
    y: 35 to 85

Intersection:

    width  = 60 - 35 = 25
    height = 60 - 35 = 25
    area   = 25 * 25 = 625

Areas:

    area_A = 50 * 50 = 2500
    area_B = 50 * 50 = 2500

Union:

    union = 2500 + 2500 - 625 = 4375

IoU:

    IoU = 625 / 4375 = 0.142857

Rounded:

    0.1429

---

# 7. Success Curve and AUC

## Code

    def compute_success_curve(
        ious: np.ndarray,
        n_thresholds: int = 21
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Defect C fix:
        # OPE success curve uses 21 thresholds from 0.0 to 1.0.
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        success = np.array([
            np.mean(ious >= t)
            for t in thresholds
        ])

        return thresholds, success


    def compute_auc(thresholds: np.ndarray, success: np.ndarray) -> float:
        return float(
            np.trapezoid(success, thresholds)
            / (thresholds[-1] - thresholds[0])
        )

## Explanation

The success curve measures the fraction of frames where IoU is above a threshold.

For each threshold `t`:

    success(t) = number_of_frames_with_IoU >= t / total_number_of_frames

The standard OPE success curve uses 21 thresholds from 0.0 to 1.0:

    0.00, 0.05, 0.10, ..., 1.00

This is produced by:

    np.linspace(0.0, 1.0, 21)

AUC means Area Under Curve.

In this notebook, AUC is computed by numerical integration:

    np.trapezoid(success, thresholds)

Then it is normalized by the threshold range:

    thresholds[-1] - thresholds[0]

Since the range is 1.0, the result is the average success over all IoU thresholds.

---

# 8. Precision Metric

## Code

    def compute_precision(
        pred_boxes: list,
        gt_boxes: list,
        threshold_px: float = 20.0
    ) -> float:

        distances = []

        for pred, gt in zip(pred_boxes, gt_boxes):
            px = pred[0] + pred[2] / 2
            py = pred[1] + pred[3] / 2

            gx = gt[0] + gt[2] / 2
            gy = gt[1] + gt[3] / 2

            distances.append(
                np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            )

        return float(np.mean(np.array(distances) <= threshold_px))

## Explanation

Precision measures center-location accuracy.

For each frame:

1. Compute the predicted box center.
2. Compute the ground-truth box center.
3. Compute the Euclidean distance between centers.
4. Count the frame as correct if the distance is within a threshold.

Default threshold:

    20 pixels

Predicted center:

    px = x_pred + w_pred / 2
    py = y_pred + h_pred / 2

Ground-truth center:

    gx = x_gt + w_gt / 2
    gy = y_gt + h_gt / 2

Distance:

    sqrt((px - gx)^2 + (py - gy)^2)

Precision:

    fraction of frames where distance <= 20 pixels

---

# 9. Stage 1 Verification

## Code

    frame_files, gt_boxes = load_sequence(DATASET_PATH, SEQUENCES[0])

    basenames = [
        os.path.basename(f)
        for f in frame_files[:3]
    ]

    assert basenames == sorted(basenames), \
        f'FAIL: frames not in order: {basenames}'

    print('Frame order OK:', basenames[:3])

    boxA = (10.0, 10.0, 50.0, 50.0)
    boxB = (35.0, 35.0, 50.0, 50.0)

    expected_iou = 0.1429
    computed_iou = round(compute_iou(boxA, boxB), 4)

    assert computed_iou == expected_iou, \
        f'IoU FAIL: got {computed_iou}, expected {expected_iou}'

    dummy_ious = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

    thresh, suc = compute_success_curve(dummy_ious)

    assert thresh[0] == 0.0, \
        f'Success curve FAIL: first threshold is {thresh[0]}, expected 0.0'

    print('PASS  |  IoU:', computed_iou, '  first threshold:', thresh[0])

## Output

    Frame order OK: ['00000001.jpg', '00000002.jpg', '00000003.jpg']
    PASS  |  IoU: 0.1429   first threshold: 0.0

## Explanation

The verification checks three things:

1. Frame paths are sorted.
2. IoU formula gives the expected value.
3. Success curve starts at threshold 0.0.

If any of these fail, the evaluation pipeline is not reliable.

---

# 10. Stage 2: Tracker Runner and Comparative Evaluation

## Goal

Stage 2 runs all trackers on all sequences and computes metrics.

The original tracker runner had one sabotage:

OpenCV trackers expect the initialization bounding box in this format:

    (x, y, w, h)

The sabotaged code passed:

    (x1, y1, x2, y2)

This does not crash, because OpenCV accepts the tuple, but it initializes the tracker on the wrong region.

---

# 11. Running One Tracker

## Code

    def run_tracker(
        tracker_constructor,
        frame_files: List[str],
        init_bbox: Tuple[float, float, float, float]
    ) -> List[Tuple]:

        tracker = tracker_constructor()
        frame0 = cv2.imread(frame_files[0])

        x, y, w, h = init_bbox

        # Sabotage fix:
        # OpenCV tracker.init expects (x, y, width, height),
        # not (x1, y1, x2, y2).
        init_box = (
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h))
        )

        tracker.init(frame0, init_box)

        predicted = [init_bbox]

        for path in frame_files[1:]:
            frame = cv2.imread(path)
            ok, box = tracker.update(frame)

            if ok:
                predicted.append(tuple(box))
            else:
                predicted.append((0.0, 0.0, 0.0, 0.0))

        return predicted

## Explanation

The tracker is initialized on the first frame.

The initial bounding box comes from ground truth:

    gt_boxes[0]

The tracker then updates on every following frame.

If tracking succeeds:

    ok == True

then the predicted box is stored.

If tracking fails:

    ok == False

then the notebook stores an empty box:

    (0.0, 0.0, 0.0, 0.0)

This keeps the prediction list aligned with the ground-truth list.

---

## Correct Bounding Box Format

OpenCV tracker initialization requires:

    (x, y, w, h)

where:

- x = top-left x coordinate;
- y = top-left y coordinate;
- w = width;
- h = height.

Wrong format:

    (x, y, x + w, y + h)

This represents corner coordinates, not width and height.

Why this matters:

If the tracker receives `(x, y, x + w, y + h)`, it interprets `x + w` as width and `y + h` as height.

The initialization box becomes too large and shifted in meaning.

This corrupts tracking from the first frame.

---

# 12. Running All Trackers on All Sequences

## Code

    results = {}

    for seq in SEQUENCES:
        print(f'\nSequence: {seq}')

        frame_files, gt_boxes = load_sequence(DATASET_PATH, seq)
        results[seq] = {}

        for name, constructor in TRACKERS.items():
            print(f'  Running {name}...', end=' ')

            pred = run_tracker(
                constructor,
                frame_files,
                gt_boxes[0]
            )

            results[seq][name] = pred

            print('done')

## Explanation

The dictionary `results` stores all predictions.

Structure:

    results[sequence][tracker_name] = list_of_predicted_boxes

Example:

    results['car-17']['KCF']

contains the KCF predictions for the `car-17` sequence.

Each prediction list has one bounding box per frame.

---

# 13. Computing Metrics

## Code

    metrics = {}

    for seq in SEQUENCES:
        _, gt_boxes = load_sequence(DATASET_PATH, seq)
        metrics[seq] = {}

        for name in TRACKERS:
            pred = results[seq][name]

            ious = np.array([
                compute_iou(p, g)
                for p, g in zip(pred, gt_boxes)
            ])

            thresholds, success = compute_success_curve(ious)

            metrics[seq][name] = {
                'mean_iou':  float(np.mean(ious)),
                'auc':       compute_auc(thresholds, success),
                'precision': compute_precision(pred, gt_boxes)
            }

    header = f'{"Sequence":<12} {"Tracker":<12} {"Mean IoU":>10} {"AUC":>8} {"Precision":>12}'

    print('\n' + header)
    print('-' * 58)

    for seq in SEQUENCES:
        for name in TRACKERS:
            m = metrics[seq][name]
            print(
                f'{seq:<12} {name:<12} '
                f'{m["mean_iou"]:>10.3f} '
                f'{m["auc"]:>8.3f} '
                f'{m["precision"]:>12.3f}'
            )

    csrt_aucs = [
        metrics[seq]['CSRT']['auc']
        for seq in SEQUENCES
    ]

    kcf_aucs = [
        metrics[seq]['KCF']['auc']
        for seq in SEQUENCES
    ]

    assert any(c > k for c, k in zip(csrt_aucs, kcf_aucs)), \
        'FAIL: CSRT does not outperform KCF on any sequence. Check your sabotage fix.'

    print('\nIntegration assertion PASS')

## Explanation

For each sequence and tracker, the notebook computes:

1. Mean IoU.
2. AUC.
3. Precision.

Mean IoU:

    average IoU across frames

AUC:

    area under the success curve

Precision:

    fraction of frames where center error <= 20 pixels

The integration assertion verifies that CSRT beats KCF on at least one sequence.

This is not a full correctness proof, but it catches the initialization sabotage.

---

# 14. Stage 2 Output Metrics

## Output

    Sequence     Tracker        Mean IoU      AUC    Precision
    ----------------------------------------------------------
    car-17       KCF               0.467    0.478        0.582
    car-17       CSRT              0.471    0.480        0.607
    car-17       DaSiamRPN         0.462    0.472        0.603

    car-20       KCF               0.511    0.521        0.387
    car-20       CSRT              0.357    0.369        0.430
    car-20       DaSiamRPN         0.377    0.387        0.372

    person-7     KCF               0.027    0.051        0.073
    person-7     CSRT              0.059    0.081        0.100
    person-7     DaSiamRPN         0.051    0.074        0.083

    person-17    KCF               0.204    0.221        0.147
    person-17    CSRT              0.196    0.213        0.173
    person-17    DaSiamRPN         0.317    0.324        0.202

    bicycle-18   KCF               0.023    0.047        0.018
    bicycle-18   CSRT              0.045    0.067        0.018
    bicycle-18   DaSiamRPN         0.476    0.479        0.458

    bicycle-9    KCF               0.008    0.033        0.008
    bicycle-9    CSRT              0.027    0.050        0.027
    bicycle-9    DaSiamRPN         0.047    0.069        0.055

    Integration assertion PASS

---

# 15. Metrics Table

| Sequence   | Tracker   | Mean IoU |   AUC | Precision |
| ---------- | --------- | -------: | ----: | --------: |
| car-17     | KCF       |    0.467 | 0.478 |     0.582 |
| car-17     | CSRT      |    0.471 | 0.480 |     0.607 |
| car-17     | DaSiamRPN |    0.462 | 0.472 |     0.603 |
| car-20     | KCF       |    0.511 | 0.521 |     0.387 |
| car-20     | CSRT      |    0.357 | 0.369 |     0.430 |
| car-20     | DaSiamRPN |    0.377 | 0.387 |     0.372 |
| person-7   | KCF       |    0.027 | 0.051 |     0.073 |
| person-7   | CSRT      |    0.059 | 0.081 |     0.100 |
| person-7   | DaSiamRPN |    0.051 | 0.074 |     0.083 |
| person-17  | KCF       |    0.204 | 0.221 |     0.147 |
| person-17  | CSRT      |    0.196 | 0.213 |     0.173 |
| person-17  | DaSiamRPN |    0.317 | 0.324 |     0.202 |
| bicycle-18 | KCF       |    0.023 | 0.047 |     0.018 |
| bicycle-18 | CSRT      |    0.045 | 0.067 |     0.018 |
| bicycle-18 | DaSiamRPN |    0.476 | 0.479 |     0.458 |
| bicycle-9  | KCF       |    0.008 | 0.033 |     0.008 |
| bicycle-9  | CSRT      |    0.027 | 0.050 |     0.027 |
| bicycle-9  | DaSiamRPN |    0.047 | 0.069 |     0.055 |

---

# 16. Plotting Success Curves

## Code

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, seq in zip(axes, SEQUENCES):
        _, gt_boxes = load_sequence(DATASET_PATH, seq)

        for name, color in zip(
            TRACKERS,
            ['steelblue', 'darkorange', 'seagreen']
        ):
            pred = results[seq][name]

            ious = np.array([
                compute_iou(p, g)
                for p, g in zip(pred, gt_boxes)
            ])

            thresholds, success = compute_success_curve(ious)
            auc = compute_auc(thresholds, success)

            ax.plot(
                thresholds,
                success,
                label=f'{name} (AUC={auc:.2f})',
                color=color
            )

        ax.axhline(
            0.5,
            color='gray',
            linestyle='--',
            linewidth=0.8
        )

        ax.set_title(seq)
        ax.set_xlabel('IoU threshold')
        ax.set_ylabel('Success rate')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Success Curves — OPE Protocol', fontsize=12)
    plt.tight_layout()
    plt.show()

## Explanation

This plot shows one success curve per tracker for each sequence.

X-axis:

    IoU threshold

Y-axis:

    fraction of successful frames

A better tracker has a curve that stays higher for more thresholds.

AUC summarizes the whole success curve into one number.

---

# 17. Plotting IoU Over Time

## Code

    seq = 'bicycle-18'

    _, gt_boxes = load_sequence(DATASET_PATH, seq)

    fig, ax = plt.subplots(figsize=(12, 3))

    for name, color in zip(
        TRACKERS,
        ['steelblue', 'darkorange', 'seagreen']
    ):
        ious = np.array([
            compute_iou(p, g)
            for p, g in zip(results[seq][name], gt_boxes)
        ])

        ax.plot(
            ious,
            label=name,
            color=color,
            alpha=0.8,
            linewidth=0.8
        )

    ax.axhline(
        0.5,
        color='gray',
        linestyle='--',
        linewidth=0.8,
        label='IoU=0.5'
    )

    ax.set_xlabel('Frame')
    ax.set_ylabel('IoU')
    ax.set_title(f'IoU over time — {seq}')
    ax.legend()

    plt.tight_layout()
    plt.show()

## Explanation

The IoU-over-time plot shows how tracking quality changes frame by frame.

This is useful because a single AUC value hides temporal behavior.

A tracker may:

- work well at the beginning;
- fail after occlusion;
- drift slowly;
- recover later;
- never recover after target loss.

For `bicycle-18`, the results show a large gap between DaSiamRPN and the classical correlation-filter trackers.

---

# 18. Stage 3: Failure Mode Analysis

## Goal

A tracker is defined as failed when:

1. Its IoU drops below 0.2.
2. It does not recover above 0.2 within the next 10 frames.

Stage 3 finds the first failure onset for each tracker on:

    person-17

Then it visualises 5 frames around each failure:

- 2 frames before failure;
- failure frame;
- 2 frames after failure.

Predicted box:

    red

Ground-truth box:

    green

---

# 19. Finding Failure Onset

## Code

    def find_failure_onset(
        ious: np.ndarray,
        iou_threshold: float = 0.2,
        recovery_window: int = 10
    ) -> int:

        for i in range(len(ious)):
            if ious[i] < iou_threshold:
                future_window = ious[i + 1 : i + recovery_window + 1]
                recovered = np.any(future_window >= iou_threshold)

                if not recovered:
                    return i

        return -1

## Explanation

The function scans the IoU sequence from the beginning.

For each frame `i`:

1.  Check if IoU is below the failure threshold:

        ious[i] < 0.2

2.  Look at the next `recovery_window` frames:

        ious[i + 1 : i + recovery_window + 1]

3.  If none of those future frames recover above the threshold, the tracker is considered failed at frame `i`.

4.  If no such frame exists, the function returns:

        -1

This avoids treating short temporary drops as permanent failures.

---

# 20. Visualising Failure

## Code

    def visualise_failure(
        sequence: str,
        tracker_name: str,
        frame_files: List[str],
        pred_boxes: list,
        gt_boxes: list,
        failure_frame: int,
        context: int = 2
    ):

        start = max(0, failure_frame - context)
        end   = min(len(frame_files), failure_frame + context + 1)
        n     = end - start

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        if n == 1:
            axes = [axes]

        for ax, idx in zip(axes, range(start, end)):
            img = cv2.cvtColor(
                cv2.imread(frame_files[idx]),
                cv2.COLOR_BGR2RGB
            )

            ax.imshow(img)

            px, py, pw, ph = pred_boxes[idx]
            gx, gy, gw, gh = gt_boxes[idx]

            ax.add_patch(
                patches.Rectangle(
                    (px, py),
                    pw,
                    ph,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    label='Predicted'
                )
            )

            ax.add_patch(
                patches.Rectangle(
                    (gx, gy),
                    gw,
                    gh,
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none',
                    label='Ground truth'
                )
            )

            iou_val = compute_iou(
                pred_boxes[idx],
                gt_boxes[idx]
            )

            marker = ' *** FAILURE' if idx == failure_frame else ''

            ax.set_title(
                f'Frame {idx}\nIoU={iou_val:.2f}{marker}',
                fontsize=8
            )

            ax.axis('off')

        axes[0].legend(loc='upper left', fontsize=7)
        fig.suptitle(f'{tracker_name} on {sequence}', fontsize=10)

        plt.tight_layout()
        plt.show()

## Explanation

This function displays a small temporal window around the failure frame.

For each frame, it draws:

- red rectangle = predicted box;
- green rectangle = ground truth.

It also prints the IoU for that frame.

The failure frame title includes:

    *** FAILURE

This visualisation helps diagnose whether the tracker failed due to:

- drift;
- occlusion;
- scale change;
- similar distractor;
- target leaving the search window;
- template becoming outdated.

---

# 21. Running Failure Analysis

## Code

    seq = 'person-17'

    frame_files, gt_boxes = load_sequence(DATASET_PATH, seq)

    failure_frames = {}

    print(f'Failure onset analysis — {seq}')

    header = f'{"Tracker":<12}  {"Failure frame":>15}  {"IoU at failure":>16}'
    print(header)
    print('-' * 48)

    for name in TRACKERS:
        ious = np.array([
            compute_iou(p, g)
            for p, g in zip(results[seq][name], gt_boxes)
        ])

        failure_frame = find_failure_onset(ious)
        failure_frames[name] = failure_frame

        iou_at_fail = (
            ious[failure_frame]
            if failure_frame >= 0
            else float('nan')
        )

        print(
            f'{name:<12}  '
            f'{str(failure_frame):>15}  '
            f'{iou_at_fail:>16.3f}'
        )

        if failure_frame >= 0:
            visualise_failure(
                seq,
                name,
                frame_files,
                results[seq][name],
                gt_boxes,
                failure_frame
            )

## Output

    Failure onset analysis — person-17
    Tracker         Failure frame    IoU at failure
    ------------------------------------------------
    KCF                       143             0.000
    CSRT                      176             0.103
    DaSiamRPN                 296             0.000

## Explanation

The first permanent failure frames on `person-17` are:

| Tracker   | Failure frame | IoU at failure |
| --------- | ------------: | -------------: |
| KCF       |           143 |          0.000 |
| CSRT      |           176 |          0.103 |
| DaSiamRPN |           296 |          0.000 |

Interpretation:

- KCF fails first.
- CSRT survives longer than KCF on this sequence.
- DaSiamRPN survives much longer, but eventually also fails.

This supports the idea that deep Siamese trackers can be more robust than classical correlation-filter trackers under larger appearance changes or more challenging motion.

---

# 22. Concept: KCF Failure Mechanism

KCF is based on correlation filters.

It searches for the target in a window around the previous location.

Main assumptions:

- The target remains near the previous position.
- The target appearance does not change too drastically.
- The search window contains the target.
- The circular correlation assumption is acceptable.

## Circular Correlation Assumption

KCF uses cyclic shifts of the search window to train and evaluate the filter efficiently.

This makes computation fast with FFT, but it introduces a circular boundary assumption.

The search region is treated as if its left/right and top/bottom borders wrap around.

Problem:

If the target leaves the search window or is heavily occluded, the filter may update using background instead of the target.

Then the tracker starts learning the wrong appearance.

This is called drift.

## Drift

Drift happens when the tracker gradually adapts to the wrong region.

In KCF:

1. The target becomes partially occluded or moves out of the search window.
2. The maximum response may occur on background.
3. The tracker updates the correlation filter with that incorrect patch.
4. The appearance model becomes contaminated.
5. Future predictions follow the wrong region.

---

# 23. Concept: CSRT vs KCF

CSRT improves over KCF by using:

- channel reliability;
- spatial reliability maps.

Spatial reliability maps down-weight unreliable background areas.

This helps under partial occlusion because the tracker can focus on more reliable target regions.

However, CSRT still uses a local search and an online-updated appearance model.

It can still fail when:

- the target is fully occluded;
- the target leaves the search region;
- the appearance changes too much;
- the model updates on wrong samples.

CSRT is more robust than KCF under partial occlusion, but it is not guaranteed to outperform deep trackers such as DaSiamRPN under complete occlusion or major appearance change.

---

# 24. Concept: DaSiamRPN Failure Mechanism

DaSiamRPN is a Siamese-network tracker.

It uses:

- a template from the first frame;
- a search region from the current frame;
- cross-correlation between template and search features;
- a region proposal network for classification and bounding box regression.

Strengths:

- strong learned representation;
- better robustness than classical trackers;
- good scale and location estimation;
- no online drift from bad updates because the template is fixed.

Weaknesses:

- fixed template can become outdated;
- large appearance changes can make the template unrepresentative;
- similar distractors may create wrong response peaks;
- target may move outside the search region.

Because DaSiamRPN uses a fixed template, it avoids online template contamination.

But if the object appearance changes too much compared with frame 1, the fixed template can become a weakness.

---

# 25. Stage 4 Concepts

The notebook’s final report asks about three conceptual topics.

Answer cells can be skipped for the knowledge base, but the concepts are important.

---

## 25.1 AUC vs Published Benchmarks

Published OTB-2015 values mentioned in the notebook:

| Tracker | OTB-2015 AUC |
| ------- | -----------: |
| KCF     |        0.477 |
| CSRT    |        0.661 |

The lab evaluates on LaSOT-style sequences.

LaSOT differs from OTB-2015 because LaSOT contains:

- longer videos;
- higher resolution;
- more challenging objects;
- greater diversity;
- more long-term tracking failures.

This means LaSOT AUC values can be systematically lower than OTB-2015 values.

Important limitation:

OPE does not reset the tracker after failure.

A tracker that fails early and never recovers can get very low AUC on long sequences.

---

## 25.2 Template Update Problem

DaSiamRPN uses a fixed template from frame 1.

A simple template update strategy could replace the template every `k` frames.

This can help if the target appearance changes gradually.

But it can make tracking worse if the current predicted crop is wrong.

This is called template contamination.

Template contamination happens when the tracker updates its appearance model using:

- background;
- occluder;
- distractor;
- wrong object;
- inaccurate bounding box.

Once the template is contaminated, the tracker may continue matching the wrong object.

---

## 25.3 Deployment Constraint

For smart camera deployment, tracker choice depends on constraints such as:

- real-time FPS requirement;
- CPU vs GPU availability;
- power budget;
- robustness requirement;
- need for long-term recovery;
- tolerance for drift;
- memory limitations.

General trade-off:

| Tracker              | Speed       | Robustness | Notes                              |
| -------------------- | ----------- | ---------- | ---------------------------------- |
| MOSSE                | Very high   | Low        | Good for ultra-low latency         |
| KCF                  | High        | Medium-low | Fast but no scale adaptation       |
| CSRT                 | Lower       | Medium     | More robust but heavier            |
| DaSiamRPN            | High on GPU | Higher     | Better learned matching            |
| Transformer trackers | Medium-low  | High       | More expensive                     |
| DiMP / ATOM          | Lower       | High       | Strong but computationally heavier |

---

# 26. Common Bugs and Fixes

| Bug                                         | Why it is wrong                                               | Correct fix                         |
| ------------------------------------------- | ------------------------------------------------------------- | ----------------------------------- |
| `glob.glob()` without sorting               | Frames may be processed out of order                          | Use `sorted(glob.glob(...))`        |
| Wrong IoU union formula                     | Intersection is counted twice                                 | `union = area_A + area_B - inter`   |
| Success thresholds start at 0.1             | AUC calculation is shifted                                    | Use `np.linspace(0.0, 1.0, 21)`     |
| Tracker initialized with `(x1, y1, x2, y2)` | OpenCV expects width and height, not bottom-right coordinates | Use `(x, y, w, h)`                  |
| Not storing failed frames                   | Prediction and ground-truth lists become misaligned           | Append `(0, 0, 0, 0)` on failure    |
| Looking only at mean IoU                    | Hides temporal failures                                       | Plot IoU over time                  |
| Treating short IoU dip as failure           | Some trackers recover quickly                                 | Require no recovery within a window |

---

# 27. Key Formulas

## IoU

    IoU = intersection_area / union_area

    union_area = area_A + area_B - intersection_area

---

## Center Precision

    center_x = x + w / 2
    center_y = y + h / 2

    distance = sqrt((px - gx)^2 + (py - gy)^2)

    precision = fraction of frames where distance <= threshold_px

---

## Success Curve

    success(t) = mean(IoU >= t)

where:

    t in [0.0, 1.0]

with 21 thresholds:

    0.00, 0.05, 0.10, ..., 1.00

---

## AUC

    AUC = area under success curve

In code:

    auc = np.trapezoid(success, thresholds) / (thresholds[-1] - thresholds[0])

---

## Failure Onset

A failure starts at frame `i` if:

    IoU[i] < 0.2

and there is no recovery in the next 10 frames:

    no IoU[j] >= 0.2 for j in i+1 ... i+10

---

# 28. Key Takeaways

## Evaluation

Tracking evaluation is sensitive to silent bugs.

Correct evaluation requires:

- sorted frames;
- aligned ground-truth boxes;
- correct bounding box format;
- correct IoU formula;
- correct success thresholds.

---

## Metrics

Mean IoU shows average overlap.

AUC summarizes the success curve over IoU thresholds.

Precision measures center-location accuracy.

IoU-over-time plots reveal when and how a tracker fails.

---

## Trackers

KCF is fast but can drift under occlusion or when the target leaves the search window.

CSRT is more robust than KCF under partial occlusion because it uses spatial reliability.

DaSiamRPN is more robust on difficult sequences because it uses deep template matching and region proposals.

However, DaSiamRPN can still fail when the fixed template becomes unrepresentative, a distractor appears, or the object leaves the search region.

---

## Failure Analysis

Failure analysis should not rely only on final metrics.

A good analysis should inspect frames around failure onset and identify the mechanism:

- drift;
- occlusion;
- template mismatch;
- distractor;
- fast motion;
- scale change;
- search window failure.

---

## Practical Rule

For object tracking experiments:

1. Verify frame order first.
2. Verify bounding box format.
3. Verify IoU on a simple manual example.
4. Plot success curves.
5. Plot IoU over time.
6. Inspect failure frames visually.
7. Compare trackers using both metrics and failure mechanisms.

# Lab 04: The PointNet Autopsy

## 1. Lab Goal

This notebook studies PointNet for 3D point cloud processing.

The lab covers three main tasks:

1. Point cloud classification on ModelNet10.
2. Chair part segmentation on ShapeNetPart-style chair data.
3. Chamfer distance implementation for comparing point clouds.

The notebook is structured as a debugging pipeline. Each stage contains defects or sabotages that must be fixed before the verification cell passes.

Main topics:

- point cloud loading;
- point sampling;
- point cloud normalization;
- PointNet encoder;
- permutation invariance;
- max pooling as global aggregation;
- failure analysis with confusion matrices;
- segmentation using global-local feature fusion;
- Chamfer distance.

Dataset:

    uvigo-pointnet-lab04

Subfolders:

    modelnet10_hdf5/   -> point cloud classification
    chair_seg/         -> chair part segmentation
    weights/           -> pretrained PointNet weights

Hardware:

    T4 GPU recommended

---

# 2. Setup

## Code

    import os, h5py, random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from typing import Tuple, List

    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED        = 42
    N_POINTS    = 1024
    BATCH_SIZE  = 32
    NUM_CLASSES = 10
    NUM_PARTS   = 4

    PART_NAMES  = ['seat', 'armrest', 'leg', 'back']

    DATASET_PATH = '/kaggle/input/datasets/ivnrodrguezconde/uvigo-3d-vision-lab-04/uvigo-pointnet-lab04'
    WEIGHTS_PATH = os.path.join(DATASET_PATH, 'weights')

    CLASS_NAMES = [
        'bathtub',
        'bed',
        'chair',
        'desk',
        'dresser',
        'monitor',
        'night_stand',
        'sofa',
        'table',
        'toilet'
    ]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f'Device: {DEVICE}')

## Explanation

The lab uses PyTorch for neural networks and HDF5 files for ModelNet10 point clouds.

Important constants:

| Name          | Meaning                                        |
| ------------- | ---------------------------------------------- |
| `N_POINTS`    | Number of sampled points per shape             |
| `BATCH_SIZE`  | Batch size used by DataLoaders                 |
| `NUM_CLASSES` | Number of ModelNet10 classes                   |
| `NUM_PARTS`   | Number of chair segmentation labels            |
| `PART_NAMES`  | Names of chair parts after label normalization |

Point cloud classification uses:

    [B, N, 3]

where:

- `B` = batch size;
- `N` = number of points;
- `3` = x, y, z coordinates.

---

# 3. Stage 1: Data Pipeline

## Goal

The dataset must return point clouds in the correct format for PointNet.

Each point cloud should be:

1. randomly sampled;
2. centered around the origin;
3. normalized into the unit sphere;
4. returned as a tensor with shape `[N_POINTS, 3]`.

The original dataset contains three defects.

---

## Stage 1 Defects

| Defect | Problem                                      | Correct Fix                          |
| ------ | -------------------------------------------- | ------------------------------------ |
| A      | Always returns points in the same file order | Randomly sample points on every call |
| B      | Normalizes by maximum absolute coordinate    | Normalize by maximum L2 norm         |
| C      | Training DataLoader allows small final batch | Use `drop_last=True` for training    |

---

## Corrected Code

    class ModelNet10Dataset(Dataset):
        def __init__(self, root, split='train', n_points=N_POINTS):
            self.n_points = n_points

            points, labels = [], []
            pattern = 'train' if split == 'train' else 'test'

            for fname in sorted(os.listdir(root)):
                if fname.endswith('.h5') and pattern in fname:
                    with h5py.File(os.path.join(root, fname), 'r') as f:
                        points.append(f['data'][:])
                        labels.append(f['label'][:])

            self.points = np.concatenate(points, axis=0).astype(np.float32)
            self.labels = np.concatenate(labels, axis=0).astype(np.int64).squeeze()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            pts = self.points[idx]

            # Defect A fix:
            # Randomly sample N_POINTS from the shape.
            choice = np.random.choice(len(pts), self.n_points, replace=False)
            pts = pts[choice]

            # Center around the origin.
            pts = pts - pts.mean(axis=0)

            # Defect B fix:
            # Normalize by maximum L2 norm so the shape fits inside the unit sphere.
            max_norm = np.max(np.linalg.norm(pts, axis=1))
            pts = pts / max_norm

            return torch.from_numpy(pts), self.labels[idx]


    mn10_dir = os.path.join(DATASET_PATH, 'modelnet10_hdf5')

    train_dataset = ModelNet10Dataset(mn10_dir, 'train')
    test_dataset  = ModelNet10Dataset(mn10_dir, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    print(f'Dataset loaded — train: {len(train_dataset):,}  test: {len(test_dataset):,}')
    print(f'Batch size: {BATCH_SIZE}  Batches per epoch: {len(train_loader)}')

---

## Explanation: Random Sampling

Point clouds are unordered sets.

PointNet should not depend on the original file order of points.

Wrong version:

    choice = np.arange(self.n_points)

This always returns the first `N_POINTS` points in the same order.

Correct version:

    choice = np.random.choice(len(pts), self.n_points, replace=False)

This randomly samples points every time `__getitem__` is called.

Why this matters:

- It acts as data augmentation.
- It prevents the model from relying on fixed point order.
- It tests PointNet’s permutation invariance assumption.

---

## Explanation: Unit Sphere Normalization

The point cloud is first centered:

    pts = pts - pts.mean(axis=0)

Then normalized:

    pts = pts / np.max(np.linalg.norm(pts, axis=1))

The L2 norm of a point is:

    sqrt(x^2 + y^2 + z^2)

The maximum L2 norm gives the farthest point from the origin.

After division, all points lie inside the unit sphere:

    max(||p_i||_2) <= 1

Wrong version:

    pts = pts / np.max(np.abs(pts))

This normalizes by the largest coordinate value, not by the farthest 3D point. It can make scale inconsistent across shapes.

---

## Explanation: `drop_last=True`

BatchNorm expects stable batch statistics during training.

If the final batch is smaller than `BATCH_SIZE`, BatchNorm can behave inconsistently.

Wrong version:

    drop_last=False

Correct training version:

    drop_last=True

This removes the final incomplete batch during training.

For testing, `drop_last=False` is acceptable because we usually want to evaluate every sample.

---

## Stage 1 Verification Code

    # Test Defect A: same shape loaded twice must differ
    s1 = train_dataset[0][0].numpy()
    s2 = train_dataset[0][0].numpy()

    assert not np.allclose(s1, s2), \
        'FAIL Defect A: same shape returned identical points on two calls'

    # Test Defect B: all shapes must lie within unit sphere
    for i in random.sample(range(len(train_dataset)), 50):
        pts = train_dataset[i][0].numpy()
        max_norm = np.max(np.linalg.norm(pts, axis=1))

        assert max_norm <= 1.0 + 1e-5, \
            f'FAIL Defect B: shape {i} has max norm {max_norm:.4f} > 1'

    # Test Defect C: all training batches must have exactly BATCH_SIZE samples
    batch_sizes = [b[0].shape[0] for b in train_loader]

    assert all(s == BATCH_SIZE for s in batch_sizes), \
        f'FAIL Defect C: found batch sizes {set(batch_sizes)}, expected {{{BATCH_SIZE}}}'

    print('Stage 1 PASS — all three defects fixed')
    print(f'  Train: {len(train_dataset):,} shapes   Test: {len(test_dataset):,} shapes')

---

# 4. Stage 2: PointNet Classifier

## Goal

Stage 2 implements the PointNet classifier.

PointNet must satisfy two key properties:

1. Each point is processed independently by the same shared MLP.
2. The global shape descriptor is permutation-invariant.

The original encoder has two sabotages.

---

## Stage 2 Sabotages

| Sabotage | Problem                                        | Correct Fix                           |
| -------- | ---------------------------------------------- | ------------------------------------- |
| A        | `nn.Linear` is applied to flattened `[B, N*3]` | Use `nn.Conv1d(3, 64, kernel_size=1)` |
| B        | Global feature uses mean pooling               | Use max pooling over points           |

---

## Corrected PointNet Encoder and Classifier

    class PointNetEncoder(nn.Module):
        """
        Shared MLP over points.

        Input:
            x: [B, N, 3]

        Returns:
            global_feat: [B, D]
            local_feat:  [B, 64, N]
        """
        def __init__(self, feat_dim=1024):
            super().__init__()

            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, feat_dim, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(feat_dim)

            self.feat_dim = feat_dim

        def forward(self, x):
            # x: [B, N, 3]
            x_t = x.transpose(2, 1)                 # [B, 3, N]

            f1 = F.relu(self.bn1(self.conv1(x_t)))  # [B, 64, N]
            f2 = F.relu(self.bn2(self.conv2(f1)))   # [B, 128, N]
            f3 = F.relu(self.bn3(self.conv3(f2)))   # [B, feat_dim, N]

            # Max pooling over the point dimension.
            global_feat = torch.max(f3, dim=2)[0]   # [B, feat_dim]

            return global_feat, f1


    class PointNetClassifier(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES, feat_dim=1024):
            super().__init__()

            self.encoder = PointNetEncoder(feat_dim)

            self.fc1 = nn.Linear(feat_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)

            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

            self.drop = nn.Dropout(p=0.3)

        def forward(self, x):
            global_feat, _ = self.encoder(x)

            x = F.relu(self.bn1(self.fc1(global_feat)))
            x = self.drop(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)

            return x

---

## Explanation: Why `Conv1d(kernel_size=1)` Implements a Shared MLP

Input point cloud:

    [B, N, 3]

After transpose:

    [B, 3, N]

A 1D convolution with kernel size 1:

    nn.Conv1d(3, 64, 1)

applies the same learned transformation independently to every point.

For each point:

    [x, y, z] -> 64-dimensional feature

Because the kernel size is 1, the layer does not mix neighboring points.

This preserves point independence.

---

## Why Flattening Is Wrong

Wrong idea:

    x_t.reshape(B, -1)
    nn.Linear(N_POINTS * 3, 64)

This converts the point cloud into:

    [B, N * 3]

Problem:

- all points are mixed together immediately;
- the model becomes sensitive to point order;
- the network no longer treats the input as a set;
- the architecture no longer matches PointNet.

PointNet must process each point independently before global aggregation.

---

## Explanation: Why Max Pooling Is Used

Point clouds are unordered.

The final global descriptor must not change if the point order changes.

Max pooling over the point dimension:

    torch.max(f3, dim=2)[0]

produces a global feature vector:

    [B, feat_dim]

This operation is permutation-invariant.

If the points are shuffled, the maximum value in each feature channel stays the same.

Max pooling also selects the most activated point per feature dimension.

These selected points are often called critical points.

---

## Why Mean Pooling Is Not the Intended PointNet Aggregation

Mean pooling is also permutation-invariant, but it changes the PointNet behavior.

Mean pooling averages all point features:

    torch.mean(f3, dim=2)

This can dilute important geometric evidence.

Max pooling is designed to preserve the strongest evidence for each learned feature.

Example:

If one point strongly activates a feature for “chair leg endpoint,” max pooling keeps that activation.

Mean pooling may reduce it because most points do not contain that feature.

---

## Stage 2 Verification Code

    cls_model = PointNetClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    cls_model.load_state_dict(
        torch.load(
            os.path.join(WEIGHTS_PATH, 'pointnet_cls_best.pth'),
            map_location=DEVICE
        )
    )

    cls_model.eval()

    # Shape assertion
    pts_b, lbl_b = next(iter(test_loader))
    pts_b = pts_b.to(DEVICE)

    with torch.no_grad():
        out = cls_model(pts_b)

    assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
        f'FAIL: expected ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}'

    # Permutation invariance:
    # Same shape with different point order should produce same output.
    pts_single = pts_b[:1]
    perm = torch.randperm(N_POINTS)
    pts_perm = pts_single[:, perm, :]

    with torch.no_grad():
        out_orig = cls_model(pts_single)
        out_perm = cls_model(pts_perm)

    assert torch.allclose(out_orig, out_perm, atol=1e-4), \
        'FAIL: model output changed after permuting points — max pooling not applied'

    # Accuracy spot-check
    with torch.no_grad():
        sample_preds = cls_model(pts_b).argmax(dim=1).cpu()
        sample_labels = lbl_b
        batch_acc = (sample_preds == sample_labels).float().mean()

    assert batch_acc > 0.5, \
        f'FAIL Sabotage B: batch accuracy {batch_acc:.2f} is too low — check max pooling'

    print(f'PASS  |  output shape: {out.shape}  |  permutation invariance confirmed')

---

# 5. Stage 3: Failure Mode Analysis

## Goal

Stage 3 evaluates the trained classifier on the full test set.

It computes:

- overall accuracy;
- per-class accuracy;
- confusion matrix;
- most confused class pairs;
- visual examples of failure cases.

---

## Full Test Inference Code

    cls_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pts, labels in test_loader:
            pts = pts.to(DEVICE)

            preds = cls_model(pts).argmax(dim=1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    overall_acc = (all_preds == all_labels).mean()

    print(f'Overall test accuracy: {overall_acc:.4f}')

    print('\nPer-class accuracy:')
    print(f'  {"Class":<14}  Acc    N')

    for c, name in enumerate(CLASS_NAMES):
        mask = all_labels == c

        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
        else:
            acc = 0.0

        print(f'  {name:<14}  {acc:.3f}  {mask.sum()}')

---

## Explanation

The model is evaluated in `torch.no_grad()` mode.

This disables gradient tracking and reduces memory usage.

For each test batch:

1.  Move points to GPU.
2.  Predict class logits.
3.  Take the class with highest logit:

        argmax(dim=1)

4.  Store predictions and labels.
5.  Compute accuracy.

Overall accuracy:

    number of correct predictions / total predictions

Per-class accuracy:

    number of correct predictions in class c / number of samples in class c

---

## Confusion Matrix Code

    from sklearn.metrics import confusion_matrix
    import matplotlib.ticker as ticker

    cm = confusion_matrix(all_labels, all_preds)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(
        cm_norm,
        interpolation='nearest',
        cmap='Blues',
        vmin=0,
        vmax=1
    )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel='Predicted',
        ylabel='True',
        title='Normalised Confusion Matrix — ModelNet10'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                f'{cm_norm[i, j]:.2f}',
                ha='center',
                va='center',
                color='white' if cm_norm[i, j] > 0.6 else 'black',
                fontsize=7
            )

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    plt.show()

    print('Saved: confusion_matrix.png')

---

## Explanation

The confusion matrix shows how often each true class is predicted as each class.

Rows:

    true class

Columns:

    predicted class

Normalized confusion matrix:

    cm_norm[i, j] = cm[i, j] / sum(cm[i, :])

This means each row sums to 1.

The diagonal shows correct predictions.

Off-diagonal values show confusions.

Example:

If the row for `desk` has a high value under `table`, the model often confuses desks with tables.

---

## Failure Case Visualization Code

    def visualise_point_cloud(ax, pts, title, color='steelblue'):
        ax.scatter(
            pts[:, 0],
            pts[:, 2],
            pts[:, 1],
            s=1,
            c=color,
            alpha=0.6
        )

        ax.set_title(title, fontsize=9)
        ax.set_axis_off()


    # Find first failure for the two most confused class pairs
    confused_pairs = []

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((cm[i, j], i, j))

    confused_pairs.sort(reverse=True)
    top_pairs = confused_pairs[:2]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
        subplot_kw={'projection': '3d'}
    )

    fig.suptitle('Failure Cases — Most Confused Pairs', fontsize=12)

    for row, (count, true_cls, pred_cls) in enumerate(top_pairs):
        fail_idx = np.where(
            (all_labels == true_cls) & (all_preds == pred_cls)
        )[0]

        for col in range(min(3, len(fail_idx))):
            idx = fail_idx[col]

            pts = test_dataset[idx][0].numpy()

            title = (
                f'True: {CLASS_NAMES[true_cls]}\n'
                f'Pred: {CLASS_NAMES[pred_cls]}'
            )

            visualise_point_cloud(axes[row, col], pts, title)

    plt.tight_layout()
    plt.savefig('failure_cases.png', dpi=120)
    plt.show()

    print('Saved: failure_cases.png')

---

## Explanation

This cell identifies the most common misclassification pairs.

For example:

    true = desk, predicted = table

The code then visualizes point clouds from those failure cases.

This helps diagnose geometric ambiguity.

PointNet uses only points and global max pooling, so it can confuse objects with similar global structure.

Examples of likely confusing pairs:

- desk vs table;
- dresser vs night_stand;
- sofa vs bed;
- chair vs toilet in sparse/noisy cases.

The exact confused pairs depend on the model output.

---

## Density Experiment Code

The notebook asks to test classification accuracy with fewer points.

    for n in [1024, 512, 256, 128, 64]:
        preds_n = []
        labels_n = []

        tmp_ds = ModelNet10Dataset(
            mn10_dir,
            'test',
            n_points=n
        )

        tmp_ld = DataLoader(
            tmp_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        cls_model.eval()

        with torch.no_grad():
            for pts, lbl in tmp_ld:
                preds_n.append(
                    cls_model(pts.to(DEVICE)).argmax(1).cpu()
                )

                labels_n.append(lbl)

        acc_n = (
            torch.cat(preds_n) == torch.cat(labels_n)
        ).float().mean()

        print(f'  N={n:4d}  accuracy: {acc_n:.4f}')

---

## Explanation

The density experiment checks how many points PointNet needs to classify shapes reliably.

When `N_POINTS` decreases, the point cloud becomes sparser.

Effects of sparse point clouds:

- thin structures may disappear;
- critical points may be missing;
- local geometry becomes harder to infer;
- global shape becomes less complete.

PointNet can be robust to moderate sparsity, but accuracy usually drops when too few points remain.

---

# 6. Stage 4: Segmentation Head

## Goal

Stage 4 extends PointNet from classification to part segmentation.

Instead of predicting one class per object, the model predicts one part label per point.

Input:

    [B, N, 3]

Output:

    [B, N, NUM_PARTS]

For chair segmentation, parts are:

    seat
    armrest
    leg
    back

---

## Segmentation Head Sabotage

The original forward pass uses:

    fused = local_feat.mean(dim=2) + global_feat

This is wrong both dimensionally and semantically.

Problem:

- local point features are averaged across all points;
- per-point information is lost;
- global and local features are added instead of concatenated;
- the network no longer predicts using each point’s own local descriptor.

Correct idea:

1.  Keep local features:

        local_feat: [B, 64, N]

2.  Replicate global feature for every point:

        global_feat: [B, feat_dim]
        global_repeated: [B, feat_dim, N]

3.  Concatenate local and global features:

        fused: [B, feat_dim + 64, N]

---

## Corrected Segmentation Model

    class PointNetSegmentation(nn.Module):
        """
        PointNet segmentation model.

        Encoder backbone can be frozen during fine-tuning.

        Input:
            x: [B, N, 3]

        Output:
            logits: [B, N, num_parts]
        """
        def __init__(self, num_parts=NUM_PARTS, feat_dim=1024):
            super().__init__()

            self.encoder = PointNetEncoder(feat_dim)
            self.feat_dim = feat_dim

            self.conv1 = nn.Conv1d(feat_dim + 64, 512, 1)
            self.conv2 = nn.Conv1d(512, 256, 1)
            self.conv3 = nn.Conv1d(256, 128, 1)
            self.conv4 = nn.Conv1d(128, num_parts, 1)

            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)

        def forward(self, x):
            B, N, _ = x.shape

            global_feat, local_feat = self.encoder(x)
            # global_feat: [B, feat_dim]
            # local_feat:  [B, 64, N]

            global_repeated = global_feat.unsqueeze(2).repeat(1, 1, N)
            # global_repeated: [B, feat_dim, N]

            fused = torch.cat(
                [local_feat, global_repeated],
                dim=1
            )
            # fused: [B, feat_dim + 64, N]

            x = F.relu(self.bn1(self.conv1(fused)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))

            x = self.conv4(x)
            # x: [B, num_parts, N]

            return x.transpose(2, 1)
            # output: [B, N, num_parts]

---

## Explanation: Global-Local Fusion

For segmentation, the model needs both:

1. local point information;
2. global object context.

Local feature:

    local_feat: [B, 64, N]

This tells the model what each point looks like locally.

Global feature:

    global_feat: [B, 1024]

This tells the model what the whole object is.

The global feature is repeated for every point:

    global_feat.unsqueeze(2).repeat(1, 1, N)

This gives:

    [B, 1024, N]

Then it is concatenated with local point features:

    [B, 64, N] + [B, 1024, N] -> [B, 1088, N]

This allows each point classifier to know:

- the point’s local geometry;
- the global chair shape.

---

## Why Global Context Matters

A point on a chair leg and a point on a chair back may be locally similar in some cases.

The model needs the global descriptor to understand where the point belongs in the whole object.

Without global context, the segmentation head may confuse:

- leg vs back support;
- armrest vs seat edge;
- back vs vertical chair structure.

---

## Load Encoder Backbone and Freeze It

    seg_model = PointNetSegmentation(num_parts=NUM_PARTS).to(DEVICE)

    backbone_state = torch.load(
        os.path.join(WEIGHTS_PATH, 'pointnet_encoder_backbone.pth'),
        map_location=DEVICE
    )

    seg_model.encoder.load_state_dict(
        {
            k.replace('encoder.', ''): v
            for k, v in backbone_state.items()
        }
    )

    # Freeze encoder.
    # Only the segmentation MLP will be trained.
    for param in seg_model.encoder.parameters():
        param.requires_grad = False

    trainable = sum(
        p.numel()
        for p in seg_model.parameters()
        if p.requires_grad
    )

    frozen = sum(
        p.numel()
        for p in seg_model.parameters()
        if not p.requires_grad
    )

    print(f'Trainable params: {trainable:,}   Frozen (encoder): {frozen:,}')

---

## Explanation: Freezing the Encoder

The encoder was pretrained for point cloud classification.

Stage 4 reuses the encoder as a feature extractor.

Freezing means:

    param.requires_grad = False

Only the segmentation MLP is fine-tuned.

Advantages:

- faster training;
- less GPU memory;
- lower risk of overfitting;
- reuse of learned geometric features.

---

# 7. Chair Segmentation Dataset

## Code

    chair_dir = os.path.join(DATASET_PATH, 'chair_seg')

    from torch.utils.data import Dataset as TorchDataset


    class ChairSegDataset(TorchDataset):
        def __init__(self, root, split='train', n_points=N_POINTS):
            self.n_points = n_points

            data = np.load(
                os.path.join(root, f'chair_{split}_pts.npy')
            )

            segs = np.load(
                os.path.join(root, f'chair_{split}_seg.npy')
            )

            # Normalize labels so they start from 0.
            segs = segs - segs.min()

            self.data = data.astype(np.float32)
            self.segs = segs.astype(np.int64)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            pts = self.data[idx]
            seg = self.segs[idx]

            choice = np.random.choice(
                len(pts),
                self.n_points,
                replace=False
            )

            pts = pts[choice]
            seg = seg[choice]

            pts = pts - pts.mean(axis=0)
            pts = pts / np.max(np.linalg.norm(pts, axis=1))

            return torch.from_numpy(pts), torch.from_numpy(seg)


    seg_train_ds = ChairSegDataset(chair_dir, 'train')
    seg_test_ds  = ChairSegDataset(chair_dir, 'test')

    seg_train_ld = DataLoader(
        seg_train_ds,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )

    seg_test_ld = DataLoader(
        seg_test_ds,
        batch_size=16,
        shuffle=False,
        drop_last=False
    )

---

## Explanation

The segmentation dataset returns:

    points: [N_POINTS, 3]
    labels: [N_POINTS]

Each point has one part label.

The labels are normalized:

    segs = segs - segs.min()

This ensures labels start at 0.

For `NUM_PARTS = 4`, valid labels are:

    0, 1, 2, 3

The points are also centered and normalized into the unit sphere, like in ModelNet10.

---

# 8. Training the Segmentation Head

## Code

    seg_criterion = nn.CrossEntropyLoss()

    seg_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, seg_model.parameters()),
        lr=1e-3
    )


    def compute_miou(pred, true, num_parts=NUM_PARTS):
        ious = []

        for p in range(num_parts):
            inter = ((pred == p) & (true == p)).sum().float()
            union = ((pred == p) | (true == p)).sum().float()

            if union > 0:
                ious.append(inter / union)
            else:
                ious.append(torch.tensor(1.0))

        return torch.stack(ious).mean().item()


    print(f'{"Epoch":>6}  {"Train IoU":>10}  {"Test IoU":>10}')

    best_iou = 0.0

    for epoch in range(1, 11):
        seg_model.train()
        tr_ious = []

        for pts, segs in seg_train_ld:
            pts = pts.to(DEVICE)
            segs = segs.to(DEVICE)

            seg_optimizer.zero_grad()

            logits = seg_model(pts)

            loss = seg_criterion(
                logits.reshape(-1, NUM_PARTS),
                segs.reshape(-1)
            )

            loss.backward()
            seg_optimizer.step()

            pred = logits.argmax(2).cpu()
            true = segs.cpu()

            tr_ious.append(compute_miou(pred, true))

        seg_model.eval()
        te_ious = []

        with torch.no_grad():
            for pts, segs in seg_test_ld:
                pts = pts.to(DEVICE)
                segs = segs.to(DEVICE)

                logits = seg_model(pts)

                pred = logits.argmax(2).cpu()
                true = segs.cpu()

                te_ious.append(compute_miou(pred, true))

        tr_iou = np.mean(tr_ious)
        te_iou = np.mean(te_ious)

        best_iou = max(best_iou, te_iou)

        print(f'{epoch:>6}  {tr_iou:>10.4f}  {te_iou:>10.4f}')

    print(f'\nBest test IoU: {best_iou:.4f}')

---

## Explanation: Segmentation Loss

The model output has shape:

    [B, N, NUM_PARTS]

CrossEntropyLoss expects:

    [num_items, num_classes]

and labels:

    [num_items]

So the logits are reshaped:

    logits.reshape(-1, NUM_PARTS)

The labels are reshaped:

    segs.reshape(-1)

This treats every point as a training example.

If:

    B = 16
    N = 1024
    NUM_PARTS = 4

then:

    logits.reshape(-1, NUM_PARTS) -> [16384, 4]
    segs.reshape(-1)              -> [16384]

---

## Explanation: Mean IoU

IoU for one part is:

    intersection / union

where:

    intersection = points predicted as part p and truly part p
    union = points predicted as part p or truly part p

Mean IoU averages IoU over all parts.

This is better than raw accuracy for segmentation because some parts may be much smaller than others.

Example:

Armrests may be only a small fraction of chair points.

A model can have high point accuracy while still failing on armrests.

---

# 9. Per-Part IoU Breakdown

## Code

    seg_model.eval()

    part_ious_acc = [[] for _ in range(NUM_PARTS)]

    with torch.no_grad():
        for pts, segs in seg_test_ld:
            pts = pts.to(DEVICE)
            segs = segs.to(DEVICE)

            pred = seg_model(pts).argmax(2).cpu()
            true = segs.cpu()

            for p in range(NUM_PARTS):
                inter = ((pred == p) & (true == p)).sum().float()
                union = ((pred == p) | (true == p)).sum().float()

                if union > 0:
                    part_ious_acc[p].append((inter / union).item())

    print('\nPer-part IoU (final epoch):')

    for p, pname in enumerate(PART_NAMES):
        if part_ious_acc[p]:
            piou = np.mean(part_ious_acc[p])
        else:
            piou = float('nan')

        print(f'  Part {p} ({pname:<8}): {piou:.4f}')

    assert best_iou > 0.55, \
        f'FAIL: best IoU {best_iou:.4f} below expected threshold 0.55'

    print('Stage 4 PASS')

---

## Explanation

The per-part IoU breakdown shows which chair components are segmented well.

This matters because mean IoU can hide failure on small parts.

Typical difficulty:

| Part    | Expected Difficulty                    |
| ------- | -------------------------------------- |
| seat    | usually easier                         |
| back    | usually easier if large and vertical   |
| leg     | harder because thin structures         |
| armrest | often hardest because small and sparse |

---

# 10. Stage 5: Chamfer Distance

## Goal

Chamfer distance compares two point clouds as unordered sets.

It is commonly used when training 3D shape generators because there is usually no fixed point-to-point correspondence.

Given two point clouds:

    S1
    S2

Chamfer distance is:

    d_CD(S1, S2)
    =
    sum over x in S1 of min over y in S2 ||x - y||^2
    +
    sum over y in S2 of min over x in S1 ||x - y||^2

---

## Stage 5 Sabotages

| Sabotage | Problem                                                                      | Correct Fix            |
| -------- | ---------------------------------------------------------------------------- | ---------------------- |
| A        | Uses `argmin` and `gather` instead of differentiable nearest-distance values | Use `dist.min(...)[0]` |
| B        | Computes only one direction                                                  | Add both directions    |

---

## Correct Chamfer Distance Code

    def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
        """
        Compute Chamfer distance between two point clouds.

        Args:
            pc1: [B, N, 3]
            pc2: [B, M, 3]

        Returns:
            scalar Chamfer distance averaged over batch
        """
        # Pairwise squared distances:
        # pc1.unsqueeze(2): [B, N, 1, 3]
        # pc2.unsqueeze(1): [B, 1, M, 3]
        # diff:             [B, N, M, 3]
        diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)

        # dist: [B, N, M]
        dist = (diff ** 2).sum(dim=-1)

        # For every point in pc1, find nearest point in pc2.
        min_1_to_2 = dist.min(dim=2)[0]   # [B, N]

        # For every point in pc2, find nearest point in pc1.
        min_2_to_1 = dist.min(dim=1)[0]   # [B, M]

        # Mean version of symmetric Chamfer distance.
        term1 = min_1_to_2.mean(dim=1)
        term2 = min_2_to_1.mean(dim=1)

        return (term1 + term2).mean()

---

## Explanation: Pairwise Distance Tensor

Input shapes:

    pc1: [B, N, 3]
    pc2: [B, M, 3]

After unsqueeze:

    pc1.unsqueeze(2): [B, N, 1, 3]
    pc2.unsqueeze(1): [B, 1, M, 3]

Broadcasted difference:

    diff: [B, N, M, 3]

Squared distance:

    dist: [B, N, M]

Each value is:

    dist[b, i, j] = ||pc1[b, i] - pc2[b, j]||^2

---

## Explanation: Two Directions

First direction:

    min_1_to_2 = dist.min(dim=2)[0]

For every point in `pc1`, find the closest point in `pc2`.

Second direction:

    min_2_to_1 = dist.min(dim=1)[0]

For every point in `pc2`, find the closest point in `pc1`.

Both directions are required.

If only one direction is used, a predicted point cloud can cover only part of the target and still get a deceptively low loss.

---

## Why Chamfer Distance Is Useful

Point clouds are unordered.

There is no guaranteed correspondence such as:

    predicted point 17 corresponds to target point 17

Chamfer distance avoids this by using nearest neighbors.

It measures:

1. how close predicted points are to the target;
2. how well target points are covered by predicted points.

---

## Stage 5 Verification Code

    torch.manual_seed(0)

    # Synthetic test sets
    A = torch.randn(2, 100, 3, requires_grad=True)
    B = torch.randn(2, 120, 3)

    cd = chamfer_distance(A, B)

    # Symmetry test
    cd_ab = chamfer_distance(A.detach(), B)
    cd_ba = chamfer_distance(B, A.detach())

    assert torch.isclose(cd_ab, cd_ba, atol=1e-5), \
        f'FAIL Sabotage B: chamfer(A,B)={cd_ab:.6f} != chamfer(B,A)={cd_ba:.6f}'

    # Differentiability test
    cd_grad = chamfer_distance(A, B)
    cd_grad.backward()

    assert A.grad is not None and not torch.all(A.grad == 0), \
        'FAIL Sabotage A: gradient did not flow through chamfer distance'

    # Self-distance must be zero
    C = torch.randn(1, 50, 3)
    cd_self = chamfer_distance(C, C)

    assert torch.isclose(cd_self, torch.tensor(0.0), atol=1e-6), \
        f'FAIL: chamfer(C,C) = {cd_self:.6f}, expected 0'

    # Shifted sets should give non-zero distance
    D = C + 1.0

    cd_shift = chamfer_distance(C, D)

    assert cd_shift > 0, \
        f'FAIL: chamfer(C, C+1) should be > 0, got {cd_shift:.6f}'

    cd_shift_rev = chamfer_distance(D, C)

    assert torch.isclose(cd_shift, cd_shift_rev, atol=1e-4), \
        f'FAIL: chamfer(C,D) = {cd_shift:.6f} != chamfer(D,C) = {cd_shift_rev:.6f}'

    print(f'  chamfer(C,C+1) = {cd_shift.item():.6f}  chamfer(C+1,C) = {cd_shift_rev.item():.6f}')

    print('Stage 5 PASS')
    print(f'  chamfer(A, B)  = {cd_ab.item():.6f}')
    print(f'  chamfer(B, A)  = {cd_ba.item():.6f}')
    print(f'  chamfer(C, C)  = {cd_self.item():.6f}  (expected 0.0)')
    print(f'  chamfer(C,C+1) = {cd_shift.item():.6f}  (expected > 0)')

---

# 11. Key Shape Summary

## Classification Pipeline

| Stage                   | Shape                 |
| ----------------------- | --------------------- |
| Raw point cloud sample  | `[N_POINTS, 3]`       |
| Batch input             | `[B, N_POINTS, 3]`    |
| After transpose         | `[B, 3, N_POINTS]`    |
| First shared MLP layer  | `[B, 64, N_POINTS]`   |
| Second shared MLP layer | `[B, 128, N_POINTS]`  |
| Third shared MLP layer  | `[B, 1024, N_POINTS]` |
| Global max pooling      | `[B, 1024]`           |
| Class logits            | `[B, NUM_CLASSES]`    |

---

## Segmentation Pipeline

| Stage                        | Shape               |
| ---------------------------- | ------------------- |
| Input point cloud            | `[B, N, 3]`         |
| Local features               | `[B, 64, N]`        |
| Global feature               | `[B, 1024]`         |
| Replicated global feature    | `[B, 1024, N]`      |
| Concatenated feature         | `[B, 1088, N]`      |
| Part logits before transpose | `[B, NUM_PARTS, N]` |
| Final part logits            | `[B, N, NUM_PARTS]` |

---

## Chamfer Distance Pipeline

| Tensor                                | Shape          |
| ------------------------------------- | -------------- |
| `pc1`                                 | `[B, N, 3]`    |
| `pc2`                                 | `[B, M, 3]`    |
| `diff`                                | `[B, N, M, 3]` |
| `dist`                                | `[B, N, M]`    |
| nearest distances from `pc1` to `pc2` | `[B, N]`       |
| nearest distances from `pc2` to `pc1` | `[B, M]`       |
| final Chamfer distance                | scalar         |

---

# 12. Important Concepts for Final Report

## 12.1 Why PointNet Uses `Conv1d(kernel_size=1)`

A 1D convolution with kernel size 1 applies the same function to each point independently.

This means each point is processed without looking at neighboring points.

Geometrically, this enforces point independence.

The model learns point-wise features such as:

- “this point is at an extremity”;
- “this point lies on a flat surface”;
- “this point may belong to a leg-like part.”

Flattening the point cloud and applying `nn.Linear` breaks this property because the layer mixes all points at once and depends on point order.

---

## 12.2 Why Max Pooling Gives Permutation Invariance

Point clouds are unordered sets.

If we shuffle the points, the object is still the same.

Max pooling over the point dimension gives the same global descriptor regardless of point order.

For each feature channel, PointNet keeps the strongest point activation.

This creates a global shape descriptor based on critical points.

---

## 12.3 Why Confusions Happen

PointNet may confuse classes when their critical points are geometrically similar.

Example:

- desk vs table;
- dresser vs night_stand;
- bed vs sofa.

Reason:

PointNet’s global descriptor is built from the strongest per-feature activations.

If two classes share similar extremities, flat surfaces, rectangular shapes, or support structures, their critical points can activate similar features.

Because vanilla PointNet does not explicitly model local neighborhoods, it may miss structural relations such as:

- number of legs;
- arrangement of supports;
- local part connectivity;
- fine-grained shape details.

---

## 12.4 Why Sparse Point Clouds Increase Chamfer Loss

Chamfer distance uses nearest-neighbor search.

If a point cloud is sparse, many target points have farther nearest neighbors.

Even if the underlying continuous shape is the same, fewer sampled points means poorer surface coverage.

This increases:

    min over y in S2 ||x - y||^2

and:

    min over x in S1 ||x - y||^2

Sparse clouds lose fine geometry and produce larger nearest-neighbor gaps.

---

## 12.5 Why Segmentation Needs Global Feature Replication

For segmentation, every point needs a label.

Each point prediction should use:

- local information about that point;
- global information about the whole object.

The global descriptor is replicated N times so every point receives object-level context.

Without replication, the segmentation head cannot combine point-specific information with global shape context.

If we only average local features globally, we lose:

- which point is being classified;
- local geometric differences;
- part-level spatial cues.

---

## 12.6 Why PointNet++ Can Improve Over PointNet

PointNet processes points independently and aggregates globally.

It does not explicitly learn local neighborhoods.

PointNet++ adds hierarchical local region processing.

This helps when class differences depend on local structure.

Examples where PointNet++ may help:

- desk vs table;
- dresser vs night_stand;
- chair vs toilet;
- sofa vs bed.

Reason:

Local neighborhoods can capture part arrangements and local geometric patterns that vanilla PointNet may miss.

For example, desks and tables may both have flat tops, but their drawers, supports, and local structures differ.

PointNet++ can model such local structures more directly.

---

# 13. Common Bugs and Fixes

| Bug                                       | Why It Is Wrong                                  | Correct Fix                                               |
| ----------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| Fixed point order                         | Violates set assumption and weakens augmentation | Randomly sample points each call                          |
| Normalize by max absolute coordinate      | Does not guarantee unit sphere normalization     | Divide by max L2 norm                                     |
| `drop_last=False` during training         | Final small batch can break BatchNorm statistics | Use `drop_last=True`                                      |
| Flatten point cloud before MLP            | Destroys point independence and order invariance | Use `Conv1d(kernel_size=1)`                               |
| Mean pooling in PointNet encoder          | Dilutes critical point activations               | Use max pooling                                           |
| Averaging local features in segmentation  | Loses per-point information                      | Concatenate local features with replicated global feature |
| One-way Chamfer distance                  | Does not guarantee full shape coverage           | Use both directions                                       |
| Using `argmin` for Chamfer implementation | Uses indices instead of nearest-distance values  | Use `dist.min(...)[0]`                                    |

---

# 14. Key Takeaways

PointNet treats a point cloud as an unordered set.

The core architecture is:

    shared point-wise MLP
    max pooling
    global descriptor
    classifier or segmentation head

For classification:

    local point features -> global max pooling -> object class

For segmentation:

    local point features + replicated global feature -> per-point labels

The most important implementation details are:

- use `[B, N, 3]` as input;
- transpose to `[B, 3, N]` before `Conv1d`;
- use `Conv1d(..., kernel_size=1)` for shared per-point MLP;
- use max pooling for global aggregation;
- preserve local point features for segmentation;
- use Chamfer distance to compare unordered point sets.

PointNet is powerful but limited because it does not explicitly model local neighborhoods.

PointNet++ improves this by adding hierarchical local structure.
