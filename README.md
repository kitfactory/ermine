# Ermine

Super Easy Machine Learning Tool using TensorFlow.

## 1. What is Ermine?

Ermine is executor of the machine learning program which is composed from units.

You can edit combination of the machine learning units with excellent GUI. You can execute machine learning program like the block play. Super cool :-)

## 2.Installation

```
> pip install ermine

```

## 3.Short Example (Mnist)

At first, you have to run the server.

```
> ermine-web &

```

Open http://localhost:7007/  with your browser app.

Add mnist dataset.

Add CNN model.

Add Train Settings.

Image size & channel is important to this case.

And execute !!

You can check the progress of your traing with TensorBoard.

## 4. Custom Training.

Press trainig tab.

add units for your training.

configurate your 

Machine learning trainig process will have units bellow.

|Ermine Unit|Unit Explanation| |
|:--|:--|:--|
|Dataset Unit|some dataset preparation kit such as mnist or your custom datasets| |
|Data Augument Unit|some dataset preparation kit such as mnist or your custom datasets| |
|Model Unit| | |
|Pipline Unit|tune the machine learning input pipeline to be more faster| |
|Training Output Unit| | |

### 3.2. Inference/Test

|Ermine Unit|Unit Explanation| |
|:--|:--|:--|
|Dataset Unit|Dataset preparation unit such as mnist or your custom datasets.| |
|Inference Unit|Infer with datsets.| |
|Evaluation Output Unit|outputs result of the inferences in the specified format.|


# 4. GPU Training.

Now prinitng...

# 5. Distribute Training.

Now prinitng...

# 6. Validation of your training.

Now prinitng...


# 7. How to use ermine trained model in your program.


Ermine model can be transformed to TensorFlow Estimator. So create instance and.

# 8. Hyper parameter tuning

Ermine use Optuna.

Now prinitng...

## 5. Adding a new Ermine unit.

### 5.1. Create new unit.

### 5.2. Add the new unit to Ermine.
