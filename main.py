import matplotlib.pyplot as plt
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

from flax import nnx
from functools import partial


train_steps = 10000
eval_every = 200
batch_size = 64
learning_rate = 0.005
momentum = 0.9
tf.random.set_seed(0)


class CNNBlock(nnx.Module):
    def __init__(self, iDim, oDim, kernel_size = (3, 3), drop_out = 0.3, *, rngs: nnx.Rngs):
        # CONV RELU BN CONV RELU BN DROP POLL
        self.conv1 = nnx.Conv(iDim, oDim, kernel_size=kernel_size, padding="same", rngs=rngs)
        self.bn1 = nnx.BatchNorm(oDim, rngs=rngs)
        self.conv2 = nnx.Conv(oDim, oDim, kernel_size=kernel_size, padding="same", rngs=rngs)
        self.bn2 = nnx.BatchNorm(oDim, rngs=rngs)
        self.dropout = nnx.Dropout(drop_out, rngs=rngs)
        self.max_pool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x):
        x = self.bn1(nnx.relu(self.conv1(x)))
        x = self.bn2(nnx.relu(self.conv2(x)))
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.cnn_blocks = (
           CNNBlock(3, 64, drop_out=0.2, rngs=rngs),
           CNNBlock(64, 96, drop_out=0.3, rngs=rngs),
           CNNBlock(96, 128, drop_out=0.4, rngs=rngs),
        )
        
        self.linear1 = nnx.Linear(2048, 128, rngs=rngs)
        self.bn = nnx.BatchNorm(128, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.linear2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x):
        for block in self.cnn_blocks:
           x = block(x)

        x = x.reshape(x.shape[0], -1)

        x = nnx.relu(self.linear1(x))
        x = self.bn(x)
        x = self.dropout(x)

        x = self.linear2(x)
        return x

model = CNN(rngs=nnx.Rngs(0))

## Data
train_ds: tf.data.Dataset = tfds.load('cifar10', split='train')
test_ds: tf.data.Dataset = tfds.load('cifar10', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize the test set.

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat().shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)


## Training
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

def loss_fn(model: CNN, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics
  train_step(model, optimizer, metrics, batch)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.

    # Compute the metrics on the test set after each training epoch.
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, test_batch)

    # Log the test metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.

    print(
      f"[train]\tstep: {step}, "
      f"loss: {metrics_history['train_loss'][-1]}, "
      f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
    )
    print(
      f"[test]\tstep: {step}, "
      f"loss: {metrics_history['test_loss'][-1]}, "
      f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
    )




## After trainig
model.eval() # Switch to evaluation mode.

@nnx.jit
def pred_step(model: CNN, batch):
  logits = model(batch['image'])
  return logits.argmax(axis=1)

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(model, test_batch)

names = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
  ax.set_title(f'label={names[pred[i]]}')
  ax.axis('off')

plt.savefig('predict.png')
