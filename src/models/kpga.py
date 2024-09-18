# Copied from keras: https://github.com/keras-team/keras/blob/v3.5.0/keras/src/layers/rnn/lstm.py
import keras
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src import tree
from keras.src.backend import any_symbolic_tensors
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN
from keras.src.ops import OnesLike
from keras.src.ops import Operation
from keras import KerasTensor
from tensorflow import float32


# class Ones(Operation):
#     def call(self, shape, dtype=None):
#         return backend.numpy.ones(shape, dtype=None)
#
#     def compute_output_spec(self, x, dtype=None):
#         if dtype is None:
#             dtype = x.dtype
#         return KerasTensor(x.shape, dtype=dtype)


class MonotonicLSTMCell(Layer, DropoutRNNCell):
    """Cell class for the LSTM layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.LSTM` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the recurrent hidden state.
        dense_units: Positive integer, dimensionality of the hidden linear layers.
        direction: .
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        rectifier_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the dense layers. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.MonotonicLSTMCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.MonotonicLSTMCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)
    """

    delta_out = {"increasing": ops.identity, "decreasing": ops.negative}

    def __init__(
        self,
        units,
        dense_units,
        direction="increasing",
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        rectifier_activation="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        rectifier_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, " f"expected a positive integer, got {units}."
            )
        implementation = kwargs.pop("implementation", 2)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)

        self.unit_forget_bias = unit_forget_bias
        self.state_size = [self.units, self.units, 1]
        self.output_size = 1
        self.implementation = implementation

        # Custom variables
        self.dense_units = dense_units
        self.direction = direction

        # Custom weights
        self.delta_net_1 = keras.layers.Dense(
            self.dense_units,
            activation=rectifier_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="rectifier_1"
        )
        self.delta_net_2 = keras.layers.Dense(
            self.dense_units,
            activation=rectifier_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="rectifier_2"
        )
        self.delta_net_out = keras.layers.Dense(
            1,
            activation="relu",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="delta_out"
        )
        self.delta_transf = self.delta_out[self.direction]

        # Dropout
        self.rectifier_dropout = min(1.0, max(0.0, rectifier_dropout))
        self._m_dropout_mask = None
        self._rectifier_dropout_masks = None

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        # Kernel for the output variable
        self.z_kernel = self.add_weight(
            shape=(self.output_size, self.units * 4),
            name="z_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return ops.concatenate(
                        [
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.get("ones")((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.delta_net_1.build(input_shape[:-1] + (self.units,))
        self.delta_net_2.build(input_shape[:-1] + (self.dense_units,))
        self.delta_net_out.build(input_shape[:-1] + (self.dense_units,))

        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1, m_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        m_tm1_i, m_tm1_f, m_tm1_c, m_tm1_o = m_tm1
        i = self.recurrent_activation(
            x_i
            + ops.matmul(h_tm1_i, self.recurrent_kernel[:, : self.units])
            + ops.matmul(m_tm1_i, self.z_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f
            + ops.matmul(h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2])
            + ops.matmul(m_tm1_f, self.z_kernel[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + ops.matmul(
                h_tm1_c,
                self.recurrent_kernel[:, self.units * 2 : self.units * 3],
            )
            + ops.matmul(m_tm1_c, self.z_kernel[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o
            + ops.matmul(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
            + ops.matmul(m_tm1_o, self.z_kernel[:, self.units * 3 :])
        )
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def _create_c_dropout_mask(self, step_input, dropout_rate, count):
        ones = ops.ones_like(step_input)
        if count is None:
            return backend.random.dropout(ones, rate=dropout_rate, seed=self.seed_generator)
        else:
            return [backend.random.dropout(ones, rate=dropout_rate, seed=self.seed_generator) for _ in range(count)]

    def get_m_dropout_mask(self, step_input):
        if not hasattr(self, "_m_dropout_mask"):
            self._m_dropout_mask = None
        if self._m_dropout_mask is None and self.recurrent_dropout > 0:
            self._m_dropout_mask = self._create_c_dropout_mask(
                step_input, self.recurrent_dropout, count=None
            )
        return self._m_dropout_mask

    def reset_m_dropout_mask(self):
        self._m_dropout_mask = None

    def get_rectifier_dropout_masks(self, step_input):
        if not hasattr(self, "_rectifier_dropout_masks"):
            self._rectifier_dropout_masks = None
        if self._rectifier_dropout_masks is None and self.rectifier_dropout > 0:
            self._rectifier_dropout_masks = self._create_c_dropout_mask(
                step_input, self.rectifier_dropout, count=2
            )
        return self._rectifier_dropout_masks

    def reset_rectifier_dropout_masks(self):
        self._rectifier_dropout_masks = None

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        m_tm1 = states[2]  # previous monotonic value

        dp_mask = self.get_dropout_mask(inputs)
        # rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)
        m_dp_mask = self.get_m_dropout_mask(m_tm1)

        if training and 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask
        if training and 0.0 < self.recurrent_dropout < 1.0:
            # h_tm1 = h_tm1 * rec_dp_mask
            m_tm1 = m_tm1 * m_dp_mask

        if self.implementation == 1:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
            k_i, k_f, k_c, k_o = ops.split(self.kernel, 4, axis=1)
            x_i = ops.matmul(inputs_i, k_i)
            x_f = ops.matmul(inputs_f, k_f)
            x_c = ops.matmul(inputs_c, k_c)
            x_o = ops.matmul(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = ops.split(self.bias, 4, axis=0)
                x_i += b_i
                x_f += b_f
                x_c += b_c
                x_o += b_o

            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

            m_tm1_i = m_tm1
            m_tm1_f = m_tm1
            m_tm1_c = m_tm1
            m_tm1_o = m_tm1

            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)

            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1, (m_tm1_i, m_tm1_f, m_tm1_c, m_tm1_o))
        else:
            z = ops.matmul(inputs, self.kernel)

            z += (ops.matmul(h_tm1, self.recurrent_kernel) + ops.matmul(m_tm1, self.z_kernel))
            if self.use_bias:
                z += self.bias

            z = ops.split(z, 4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)

        delta = self.delta_net_1.call(h, training=training)
        r_masks = self.get_rectifier_dropout_masks(delta)
        if training and 0.0 < self.rectifier_dropout < 1.0:
            delta = delta * r_masks[0]

        delta = self.delta_net_2.call(delta, training=training)
        if training and 0.0 < self.rectifier_dropout < 1.0:
            delta = delta * r_masks[1]

        delta = self.delta_net_out.call(delta, training=training)
        m = m_tm1 + delta

        return m, [h, c, m]

    def get_config(self):
        config = {
            "units": self.units,
            "dense_units": self.dense_units,
            "direction": self.direction,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "rectifier_dropout": self.rectifier_dropout,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, batch_size=None):
        return [ops.zeros((batch_size, d), dtype=self.compute_dtype) for d in self.state_size]


class MonotonicLSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or backend-native)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation
    when using the TensorFlow backend.
    The requirements to use the cuDNN implementation are:

    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `dropout` == 0 and `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. Inputs, if use masking, are strictly right-padded.
    7. Eager execution is enabled in the outermost context.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> lstm = keras.layers.LSTM(4)
    >>> output = lstm(inputs)
    >>> output.shape
    (32, 4)
    >>> lstm = keras.layers.LSTM(
    ...     4, return_sequences=True, return_state=True)
    >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
    >>> whole_seq_output.shape
    (32, 10, 4)
    >>> final_memory_state.shape
    (32, 4)
    >>> final_carry_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        use_cudnn: Whether to use a cuDNN-backed implementation. `"auto"` will
            attempt to use cuDNN when feasible, and will fallback to the
            default implementation if not.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.
    """

    def __init__(
        self,
        units: int,
        dense_units: int,
        direction="increasing",
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        rectifier_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        use_cudnn=False,
        **kwargs,
    ):
        cell = MonotonicLSTMCell(
            units,
            dense_units,
            direction,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            unit_forget_bias=unit_forget_bias,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            rectifier_dropout=rectifier_dropout,
            dtype=kwargs.get("dtype", None),
            trainable=kwargs.get("trainable", True),
            name="lstm_cell",
            seed=seed,
            implementation=kwargs.pop("implementation", 2),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = InputSpec(ndim=3)
        if use_cudnn not in ("auto", True, False):
            raise ValueError(
                "Invalid valid received for argument `use_cudnn`. "
                "Expected one of {'auto', True, False}. "
                f"Received: use_cudnn={use_cudnn}"
            )
        self.use_cudnn = use_cudnn
        if (
            backend.backend() == "tensorflow"
            and backend.cudnn_ok(
                cell.activation,
                cell.recurrent_activation,
                self.unroll,
                cell.use_bias,
            )
            and use_cudnn in (True, "auto")
        ):
            self.supports_jit = False

    def inner_loop(self, sequences, initial_state, mask, training=False):
        if tree.is_nested(mask):
            mask = mask[0]

        return super().inner_loop(sequences, initial_state, mask=mask, training=training)

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(sequences, mask=mask, training=training, initial_state=initial_state)

    def _maybe_config_dropout_masks(self, cell: MonotonicLSTMCell, input_sequence, input_state):
        state = input_state[2] if isinstance(input_state, (list, tuple)) else input_state
        cell.get_dropout_mask(input_sequence)
        cell.get_m_dropout_mask(state)
        cell.reset_rectifier_dropout_masks()
        # cell.get_rectifier_dropout_masks(state) # wrong

    def _maybe_reset_dropout_masks(self, cell: MonotonicLSTMCell):
        cell.reset_dropout_mask()
        cell.reset_recurrent_dropout_mask()
        cell.reset_m_dropout_mask()
        cell.reset_rectifier_dropout_masks()

    @property
    def units(self):
        return self.cell.units

    @property
    def dense_units(self):
        return self.cell.dense_units

    @property
    def direction(self):
        return self.cell.direction

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def rectifier_dropout(self):
        return self.cell.rectifier_dropout

    def get_config(self):
        config = {
            "units": self.units,
            "dense_units": self.dense_units,
            "direction": self.direction,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "rectifier_dropout": self.rectifier_dropout,
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
