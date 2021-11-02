[This tutorial](https://github.com/tensorflow/tpu/blob/master/models/samples/core/get_started/) describes how to convert a model program using the Estimator API to one using the TPUEstimator API.

Warning The TPUEstimator API is only supported with Tensorflow 1.x. If you are writing your model with Tensorflow 2.x, use [Keras](https://keras.io/about/) instead.

## Overview

Model programs that use the TPUEstimator API can take full advantage of Tensor Processing Units (TPUs), while remaining compatible with CPUs and GPUs.

After you finish this tutorial, you will know:

- How to convert your code from using the Estimator API to using the TPUEstimator API
- How to run predictions on Cloud TPU

## Before You Begin

Before starting this tutorial, check that your Google Cloud project is correctly set up.

This walkthrough uses billable components of Google Cloud. Check the [Cloud TPU pricing page](https://cloud.google.com/tpu/docs/pricing) to estimate your costs. Be sure to [clean up](#clean_up) resources you create when you've finished with them to avoid unnecessary charges.

## Set up your resources

This section provides information on setting up Cloud Storage storage, VM, and Cloud TPU resources for tutorials.

### Create a Cloud Storage bucket

You need a Cloud Storage bucket to store the data you use to train your model and the training results. The `gcloud` command used in this tutorial sets up default permissions for the Cloud TPU service account. If you want finer-grain permissions, review the [access level permissions](https://cloud.google.com/tpu/docs/storage-buckets).

The bucket location must be in the same region as your virtual machine (VM) and your TPU node. VMs and TPU nodes are located in [specific zones](https://cloud.google.com/tpu/docs/types-zones#types), which are subdivisions within a region.

1.  Go to the Cloud Storage page on the Cloud Console.
    
    [Go to the Cloud Storage page](https://console.cloud.google.com/storage/browser)
    
2.  Create a new bucket, specifying the following options:
    
    - A unique name of your choosing.
    - Select `Region` for Location type and `us-central1` for the Location (zone)
    - Default storage class: `Standard`
    - Location: Specify a bucket location in the same region where you plan to create your TPU node. See [TPU types and zones](https://cloud.google.com/tpu/docs/types-zones#types) to learn where various TPU types are available.

### Create a TPU and VM

TPU resources are comprised of a virtual machine (VM) and a Cloud TPU that have the same name. **These resources must reside in the same region/zone as the bucket you just created.**

You can set up your VM and TPU resources using `gcloud` commands or through the [Cloud Console](https://console.cloud.google.com/). For more information about managing TPU resources, see [creating and deleting TPUs](https://cloud.google.com/tpu/docs/creating-deleting-tpus).

1.  Open a Cloud Shell window.
    
    [Open Cloud Shell](https://console.cloud.google.com/?cloudshell=true)
    
2.  Configure `gcloud` to use your project.
    
    ```
	$ gcloud config set project your-project
	```
    
    project where you want to create Cloud TPU.
3.  Launch a Compute Engine VM and Cloud TPU using the `gcloud` command.
    
    ```
	$ gcloud compute tpus execution-groups create \
     --name=tpu-name  \
     --zone=europe-west4-a \
     --tf-version=2.5.0  \
     --machine-type=n1-standard-1  \
     --accelerator-type=v3-8
	```
    
    #### Command flag descriptions
    
    `name`
    
    The name of the Cloud TPU to create.
    
    `zone`
    
    The [zone](https://cloud.google.com/tpu/docs/types-zones) where you plan to create your Cloud TPU.
    
    `tf-version`
    
    The version of Tensorflow the `gcloud` command installs on your VM.
    
    `machine-type`
    
    The [machine type](https://cloud.google.com/compute/docs/machine-types) of the Compute Engine VM to create.
    
    `accelerator-type`
    
    The [type](https://cloud.google.com/tpu/docs/types-zones) of the Cloud TPU to create.
    
    For more information on the `gcloud` command, see the [gcloud Reference](https://cloud.google.com/sdk/gcloud/reference).
    
4.  When the `gcloud compute tpus execution-groups` command has finished executing, verify that your shell prompt has changed from `username@projectname` to `username@vm-name`. This change shows that you are now logged into your Compute Engine VM.
    
    ```
	gcloud compute ssh tpu-name  --zone=europe-west4-a
	```
    

As you continue these instructions, run each command that begins with `(vm)$` in your VM session window.

### Install pandas

Install or upgrade pandas by typing the following command:

```
pip install pandas

```

## Define Hyperparameters

In this code section you add several hyperparameters that TPUs require. You add these hyperparameters as flags to your training script, which allows you to change them at runtime.

The parameters you add are:

- **tpu.** This parameter identifies the name or IP address of the TPU node on which to run the model.
- **model_dir.** The path to save model checkpoints. This path must be a Cloud Storage bucket.
- **iterations.** The number of iterations per training loop.
- **use_tpu.** Specifies if you want to run the model on TPUs or GPU/CPUs, based on availability.

### Estimator API

```python
# Model specific parameters
tf.flags.DEFINE_integer("batch_size", default=50,
    help="Batch size.")
tf.flags.DEFINE_integer("train_steps", default=1000,
    help="Total number of training steps.")FLAGS = tf.flags.FLAGS

```

### TPUEstimator API

```python
# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string( "tpu",  default=None,
    help="The Cloud TPU to use for training. This should be the name used when " "creating the Cloud TPU. To find out the name of TPU, either use command " "'gcloud compute tpus list --zone=<zone-name>', or use " "'ctpu status --details' if you have created your Cloud TPU using 'ctpu up'.")# Model specific parameters
tf.flags.DEFINE_string( "model_dir",  default="",
    help="This should be the path of storage bucket which will be used as " "model_directory to export the checkpoints during training.")
tf.flags.DEFINE_integer( "batch_size",  default=128,
    help="This is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer( "train_steps",  default=1000,
    help="Total number of training steps.")
tf.flags.DEFINE_integer( "eval_steps",  default=4,
    help="Total number of evaluation steps. If `0`, evaluation " "after training is skipped.")# TPU specific parameters.
tf.flags.DEFINE_bool( "use_tpu",  default=True,
    help="True, if want to run the model on TPU. False, otherwise.")
tf.flags.DEFINE_integer( "iterations",  default=500,
    help="Number of iterations per TPU training loop.")

```

## Loading the data

This code section specifies how to read and load the data.

TPUs support the following data types:

- `tf.float32`
- `tf.complex64`
- `tf.int64`
- `tf.bool`
- `tf.bfloat64`

### Estimator API
```python
def load_data(y_name='Species'):
  """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
  train_path, test_path = maybe_download()

  train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
  train_x, train_y = train, train.pop(y_name)

  test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
  test_x, test_y = test, test.pop(y_name)

  return (train_x, train_y), (test_x, test_y)

```
### TPUEstimator API
```python
def load_data(y_name='Species'):
  """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
  train_path, test_path = maybe_download()

  train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0,
                      dtype={'SepalLength': pd.np.float32,
                             'SepalWidth': pd.np.float32,
                             'PetalLength': pd.np.float32,
                             'PetalWidth': pd.np.float32,
                             'Species': pd.np.int32})
  train_x, train_y = train, train.pop(y_name)

  test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0,
                     dtype={'SepalLength': pd.np.float32,
                            'SepalWidth': pd.np.float32,
                            'PetalLength': pd.np.float32,
                            'PetalWidth': pd.np.float32,
                            'Species': pd.np.int32})
  test_x, test_y = test, test.pop(y_name)

  return (train_x, train_y), (test_x, test_y)

```

## Define the input functions

A key difference between the Estimator API and the TPUEstimator API is the function signature of input functions. With the Estimator API, you can write input functions with any number of parameters. With the TPUEstimator API, input functions can take only a single parameter, `params`. This `params` has all the key-value pairs from the [TPUEstimator](https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimator) object, along with extra keys like `batch_size`.

One way to address this difference is to use lambda functions when calling the input functions. With lambda functions, you need to make only minor changes to your existing input functions.

The following sections demonstrate how to update your input functions. Later, you will see how to use lambda functions to convert these input functions to work with the TPUEstimator API.

### Training input function

With the TPUEstimator API, your training input function, `train_input_fn`, must return a number of input samples that can be sharded by the number of Cloud TPU cores. For example, if you are using 8 cores, each batch size must be divisible by 8.

To accomplish this, the preceding code uses the `dataset.batch(batch_size, drop_remainder=True)` function. This function batches using the `batch_size` parameter and discards the remainder.

### Estimator API

```python
def train_input_fn(features, labels, batch_size):
  """An input function for training"""

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)

  # Return the dataset.
  return dataset

```

### TPUEstimator API

```
def train_input_fn(features, labels, batch_size):
  """An input function for training."""

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat()

  dataset = dataset.batch(batch_size, drop_remainder=True)

  # Return the dataset.
  return dataset
  
```

### Evaluation input function

In this step, you update the evaluation input function, `eval_input_fn`, to ensure that the input samples can be sharded by the number of TPU cores. To accomplish this, use the `dataset.batch(batch_size, drop_remainder=True)` function.

### Estimator API

```python
def eval_input_fn(features, labels, batch_size):
  """An input function for evaluation or prediction"""
  features=dict(features)
  if labels is None:
      # No labels, use only features.
      inputs = features
  else:
      inputs = (features, labels)

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  # Batch the examples
  assert batch_size is not None, "batch_size must not be None"
  dataset = dataset.batch(batch_size)

  # Return the dataset.
  return dataset
  
```

### TPUEstimator API

```python
 def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation."""
    features = dict(features)
    inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.shuffle(1000).repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Return the dataset.
    return dataset

```

### Prediction input function

For predictions in TPUEstimators, the input dataset must have tensors with the additional outer dimension of `batch_size`. As a result, you must add a prediction input function, which takes `features` and `batch_size` as parameters. This function allows you to have fewer input samples than `batch_size`.

A prediction input function is optional if you are using the Estimator API.

### Estimator API

> A prediction input function is optional for the Estimator API, because the evaluation function, eval_input_fn performs this task.

### TPUEstimator API

```python
  def predict_input_fn(features, batch_size):
    """An input function for prediction."""

    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(batch_size)
    return dataset

```

## Update the custom model function

Your next task is to update the custom model function:

- Replace instances of `tf.estimator.EstimatorSpec` to use `tf.contrib.tpu.TPUEstimatorSpec`.
- Remove any instances of `tf.summary`. The TPUEstimator API does not support custom summaries for tensorboard. However, basic summaries are automatically recorded to event files in the model directory.
- Wrap the optimizer using `tf.contrib.tpu.CrossShardOptimizer`. The `CrossShardOptimizer` uses an `allreduce` to aggregate gradients and broadcast the result to each shard. As the `CrossShardOptimizer` is not compatible with local training, you must also check for the `use_tpu` flag.

### Estimator API

``` python
def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""

  # Create three fully connected layers each layer having a dropout
  # probability of 0.1.
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  for units in params['hidden_units']:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

  # Compute logits (1 per class).
  logits = tf.layers.dense(net, params['n_classes'], activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'class_ids': predicted_classes[:, tf.newaxis],
          'probabilities': tf.nn.softmax(logits),
          'logits': logits,
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)

  # Compute evaluation metrics.
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predicted_classes,
                                 name='acc_op')
  metrics = {'accuracy': accuracy}
  tf.summary.scalar('accuracy', accuracy[1])
  if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN
      optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

```

### TPUEstimator API

```python
def my_model(features, labels, mode, params):
  """Deep Neural Network(DNN) model.

  This is a DNN Model with 3 hidden layers. First 2 hidden layers are having
  10 neurons in each. And number of neurons in the last layer is equal to the
  number of output classes. This is a densely connected network where each
  neuron of previous layer is connected to each neuron of next layer.

  Args:
    features: Feature values for input samples.
    labels: label/class assigned to the corresponding input sample.
    mode: "TRAIN"/"EVAL"/"PREDICT"
    params: Dictionary used to pass extra parameters to model function from
      the main function.

  Returns:
    TPUEstimatorSpec object.

  """

  # Create three fully connected layers.
  net = tf.feature_column.input_layer(features, params["feature_columns"])
  for units in params["hidden_units"]:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

  # Compute logits (1 per class).
  logits = tf.layers.dense(net, params["n_classes"], activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "class_ids": predicted_classes[:, tf.newaxis],
        "probabilities": tf.nn.softmax(logits),
        "logits": logits,
    }
    return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))

  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)

```

## Add an evaluation metric function

Another difference between the Estimator API and the TPUEstimator API is how they handle metrics. With the Estimator API, you can pass metrics as a normal dictionary. For the TPUEstimator API, you must use a function instead.

### Estimator API

> Optional. The `my_model` function generates the metrics.

### TPUEstimator API

```python
  def metric_fn(labels, logits):
    """Function to return metrics for evaluation."""

    predicted_classes = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name="acc_op")
    return {"accuracy": accuracy}

```

## Update the main function

### Configure TPUs

In this step, you configure the TPU cluster.

To configure the cluster, you can use the values assigned to the hyperparameters. See [Define Hyperparameters](#defining_hyperparameters) for more information. In addition, you must set the following values:

- `allow_soft_placement`. When set to \`true\`, this parameter allows TensorFlow to use a GPU device if a TPU is unavailable. If a GPU device is also unavailable, a CPU device is used.
- `log_device_placement`. Indicates that TensorFlow should log device placements.

### Estimator API
> Not required, as this code section only affects TPUs.

### TPUEstimator API

```python
# Resolve TPU cluster and runconfig for this.
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    FLAGS.tpu)

run_config = tf.contrib.tpu.RunConfig(
    model_dir=FLAGS.model_dir,
    cluster=tpu_cluster_resolver,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations),
)

```

### Add TPU-specific parameters to the classifier

In this section of the code, you update the classifier variable to use the TPUEstimator class. This change requires that you add the following parameters:

- `use_tpu`
- `train_batch_size`
- `eval_batch_size`
- `predict_batch_size`
- `config`

### Estimator API

```python
   # Build 2 hidden layer DNN with 10, 10 units respectively.
  classifier = tf.estimator.Estimator(
      model_fn=my_model,
      params={
          'feature_columns': my_feature_columns,
          # Two hidden layers of 10 nodes each.
          'hidden_units': [10, 10],
          # The model must choose between 3 classes.
          'n_classes': 3,
      })

```

### TPUEstimator API

```python
   # Build 2 hidden layer DNN with 10, 10 units respectively.
  classifier = tf.contrib.tpu.TPUEstimator(
      model_fn=my_model,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      config=run_config,
      params={
          # Name of the feature columns in the input data.
          "feature_columns": my_feature_columns,
          # Two hidden layers of 10 nodes each.
          "hidden_units": [10, 10],
          # The model must choose between 3 classes.
          "n_classes": 3,
          "use_tpu": FLAGS.use_tpu,
      })

```

### Call the train method

The next change is to update the train method. Notice the use of a lambda function to call the `train_input_fn` function. This methodology makes it easier to use your existing functions with the TPUEstimator API.

In addition, you must change the steps parameter to `max_steps`. In the next section, you'll repurpose the steps parameter to specify the number of evaluation steps.

### Estimator API

```python
 # Train the Model.
  classifier.train(
      input_fn=lambda:iris_data.train_input_fn(
          train_x, train_y, FLAGS.batch_size),
      steps=FLAGS.train_steps)

```

### TPUEstimator API

```python
 # Train the Model.
  classifier.train(
      input_fn=lambda  params: iris_data.train_input_fn(
          train_x, train_y,  params["batch_size"]),
      max_steps=FLAGS.train_steps)

```

### Call the evaluation method

This change is similar to the one you made to the train method. Again, the use of a lambda function makes it easier to use an existing evaluation input function.

In addition, you must change the `steps` parameter to the value set from the `eval_steps` command line flag.

### Estimator API

```python
  # Evaluate the model.
  eval_result = classifier.evaluate(
      input_fn=lambda:iris_data.eval_input_fn(
          test_x, test_y, FLAGS.batch_size))

  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

```

### TPUEstimator API

```python
  # Evaluate the model.
  eval_result = classifier.evaluate(
      input_fn=lambda params: iris_data.eval_input_fn(
          test_x, test_y, params["batch_size"]),
      steps=FLAGS.eval_steps)

```

### Call the predict method

As with the train and evaluate methods, you must update the predict method. Again, the use of a lambda function makes it easier to use an existing evaluation input function.

### Estimator API

```python
   # Generate predictions from the model
  predictions = classifier.predict(
      input_fn=lambda: iris_data.eval_input_fn(
          iris_data.PREDICTION_INPUT_DATA,
          labels=None,
          batch_size=FLAGS.batch_size))

  for pred_dict, expec in zip(predictions, iris_data.PREDICTION_OUTPUT_DATA):
      template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print(template.format(iris_data.SPECIES[class_id],
                            100 * probability, expec))

```

### TPUEstimator API

```python
   # Generate predictions from the model
  predictions = classifier.predict(
      input_fn=lambda params: iris_data.predict_input_fn(
          iris_data.PREDICTION_INPUT_DATA, params["batch_size"]))

  for pred_dict, expec in zip(predictions, iris_data.PREDICTION_OUTPUT_DATA):
    template = ("\nPrediction is \"{}\" ({:.1f}%), expected \"{}\"")

    class_id = pred_dict["class_ids"][0]
    probability = pred_dict["probabilities"][class_id]

    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))

```

## Clean up

To avoid incurring charges to your GCP account for the resources used in this topic:

1.  Disconnect from the Compute Engine VM:
    
    ```
	(vm)$ exit
    ```
	
    Your prompt should now be `username@projectname`, showing you are in the Cloud Shell.
    
2.  In your Cloud Shell, run `ctpu delete` with the --zone flag you used when you set up the Cloud TPU to delete your Compute Engine VM and your Cloud TPU:
    ```
    $ ctpu delete [optional:  --zone]
    ```

3.  Run `ctpu status` to make sure you have no instances allocated to avoid unnecessary charges for TPU usage. The deletion might take several minutes. A response like the one below indicates there are no more allocated instances:
    ```
    $ ctpu status --zone=europe-west4-a
    ```
	
```
2018/04/28 16:16:23 WARNING: Setting zone to "--zone=europe-west4-a"
No instances currently exist.
    Compute Engine VM:     --
    Cloud TPU:             --
```
	
4.  Run `gsutil` as shown, replacing bucket-name with the name of the Cloud Storage bucket you created for this tutorial:
    
	```
    $ gsutil rm -r gs://bucket-name
    ```

## What's Next?

To learn more about the Estimator and TPUEstimator APIs, see the following topics:

- Estimator API
    - [Premade Estimators](https://www.tensorflow.org/guide/premade_estimators)
    - [Creating Custom Estimators](https://www.tensorflow.org/guide/custom_estimators)
- TPUEstimator API
    - [`iris_data_tpu.ipynb`](https://github.com/tensorflow/tpu/blob/master/models/samples/core/get_started/iris_data_tpu.py)
    - [`custom_tpuestimator.ipynb`](https://github.com/tensorflow/tpu/blob/master/models/samples/core/get_started/custom_tpuestimator.py)
    - [Using TPUs](https://www.tensorflow.org/guide/using_tpu)
    - [Using the TPUEstimator API on Cloud TPUs](https://cloud.google.com/tpu/docs/using-estimator-api)