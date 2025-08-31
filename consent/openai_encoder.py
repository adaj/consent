import os
import openai
import tensorflow as tf
import numpy as np

class OpenAIEncoder(tf.keras.layers.Layer):
    def __init__(self, model_name, **kwargs):
        super(OpenAIEncoder, self).__init__(**kwargs)
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # Assuming text-embedding-3-small, which has a dimension of 1536
        self.embedding_dim = 1536

    def _get_embeddings(self, inputs):
        embeddings = []
        for text_tensor in inputs:
            text = text_tensor.numpy().decode('utf-8')[:140]
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings, dtype=np.float32)

    def call(self, inputs):
        embeddings = tf.py_function(
            self._get_embeddings,
            [inputs],
            tf.float32
        )
        embeddings.set_shape((None, self.embedding_dim))
        return embeddings

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_name": self.model_name
        })
        return config
