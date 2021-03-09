import argparse
import psutil
import tensorflow as tf
from typing import Dict, Any, Callable, Tuple

## Data Input Function
def data_input_fn(data_param,
                  batch_size:int=None,
                  shuffle=False) -> Callable[[], Tuple]:
    """Return the input function to get the test data.
    Args:
        data_param: data object
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
    Returns:
        Input function:
            - Function that returns (features, labels) when called.
    """
    
    _cpu_core_count = psutil.cpu_count(logical=False)
    
    def _input_fn() -> Tuple:
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        def map_record(record):
            return record
        
        dataset = tf.contrib.data.Dataset.from_tensor_slices(data_param)
        dataset = dataset.map(map_record, output_buffer_size=batch_size, num_threads=_cpu_core_count)
        
        if shuffle:
            # Shuffle the input unless we are predicting
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.repeat(None) # Infinite iterations: let experiment determine num_epochs
        dataset = dataset.batch(batch_size)
        
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels
    
    return _input_fn

## Model
def model(features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys, params: Dict[str, Any]):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    # Setup model architecture
    # Enable training of mode == tf.contrib.learn.ModeKeys.TRAIN
    with tf.variable_scope('Input'):
        input_layer = tf.reshape(
            features, 
            shape=[1, 1, 1], 
            name='input_reshape')

    with tf.name_scope('Dense1'):
        model_output = tf.layers.dense(
            inputs=input_layer, 
            units=10, 
            trainable=is_training)
        
        return model_output

## Model Function
# Have to remove type annotations until https://github.com/tensorflow/tensorflow/issues/12249
def custom_model_fn(features: Dict[str, tf.Tensor], 
                    labels: tf.Tensor, 
                    mode: tf.estimator.ModeKeys, 
                    params: Dict[str, Any]=None) -> tf.estimator.EstimatorSpec:
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """

    model_output = model(features, mode, params)
    
    # Get prediction of model output
    predictions = {
        'classes': tf.argmax(model_output),
        'probabilities': tf.nn.softmax(model_output, name='softmax_tensor')
    }
    
    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.softmax_cross_entropy(
        labels=tf.cast(labels, tf.int32),
        logits=model_output
    )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=tf.train.AdamOptimizer
        )
        
        # Return an EstimatorSpec object for training
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

    eval_metric = {
        'accuracy': tf.metrics.accuracy(
            labels=tf.cast(labels, tf.int32),
            predictions=model_output,
            name='accuracy'
        )
    }    
    
    # Return a EstimatorSpec object for evaluation
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric)

## Data accessors
def get_train() -> Callable[[], Tuple]:
    """Return training input_fn"""
    return data_input_fn('train', batch_size=100, shuffle=True)

def get_validation() -> Callable[[], Tuple]:
    """Return validation input_fn"""
    return data_input_fn('validation', batch_size=100)

## Experiment
def experiment_fn(run_config:tf.estimator.RunConfig, 
                  hparams:tf.contrib.training.HParams) -> tf.contrib.learn.Experiment:
    """Create an experiment to train and evaluate the model.
    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParam): Hyperparameters
    Returns:
        (Experiment) Experiment for training the mnist model.
    """
    estimator = tf.contrib.learn.Estimator(model_fn=custom_model_fn, 
                                           config=run_config,
                                           hparams=hparams)

    return tf.contrib.learn.Experiment(estimator,
                                       train_input_fn=get_train(),
                                       eval_input_fn=get_validation(),
                                       train_steps=hparams.train_steps,
                                       eval_steps=hparams.eval_steps)

def run_experiment(args):
    """Main entrypoint to run the experiment"""
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        train_steps=5000,
        eval_steps=1,
        min_eval_frequency=100
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=args.model_dir)

    schedule = 'train_and_evaluate'
    if args.train and args.evaluate:
        schedule = 'train_and_evaluate'
    elif args.train:
        schedule = 'train'
    elif args.evaluate:
        schedule = 'evaluate'

    # learn_runner will also pick up environment config...say your were running on CloudML
    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=schedule,
        hparams=params
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TF Experiment')

    parser.add_argument('--model-dir', default='./model', help='directory where checkpoints and logs will be stored')
    parser.add_argument('--data-dir', default='./data', help='directory where data is loaded or stored')

    parser.add_argument('--train', action='store_true', help='Should run training. Will run train_and_evaluate by default')
    parser.add_argument('--evaluate', action='store_true', help='Should run evaluate. Will run train_and_evaluate by default')

    args = parser.parse_args()

    # TODO: Ensure any appropriate data is downloaded so the data input_fn can use

    run_experiment(args)
