## How to install tensor2tensor (don't skip this step)
1. run command `git clone https://github.com/tensorflow/tensor2tensor.git` to clone tensor2tensor github repository.
2. run `cd tensor2tensor` to change directory to tensor2tensor.
3. go to `./tensor2tensor/bin/t2t-decoder` then 
    1. change from `import tensorflow as tf` to `import tensorflow.compat.v1 as tf`
    1. add `tf.disable_v2_behavior()` under `import tensorflow.compat.v1 as tf`
4. go to `./tensor2tensor/bin/t2t-trainer` then add `tf.disable_v2_behavior()` under `import tensorflow.compat.v1 as tf`
5. go back to the main directory of tensor2tensor then run command `pip install .` to install tensor2tensor from the cloned repository.