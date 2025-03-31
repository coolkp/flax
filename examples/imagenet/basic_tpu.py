import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

print("--- Minimal TPU jax.device_put_sharded Test (Experiment 8) ---")

try:
    devices = jax.devices('tpu')  # Explicitly request TPU devices
    print(f"  TPU Devices: {devices}") # Verify TPU devices are detected

    device_array = np.array(devices).reshape((len(devices),)) # 1D TPU mesh
    mesh = Mesh(device_array, axis_names=('data',))
    batch_sharding_tpu = NamedSharding(mesh, P('data',))

    example_array_tpu = jnp.ones((8, 224, 224, 3), dtype=jnp.float32) # Example JAX array

    shards_list_tpu = list(example_array_tpu.reshape(mesh.shape['data'], -1, 224, 224, 3)) # Split for TPU mesh

    print(f"  Number of TPU shards created: {len(shards_list_tpu)}")

    sharded_example_tpu = jax.device_put_sharded(batch_sharding_tpu, shards_list_tpu) # device_put_sharded on TPU

    print(f"  TPU jax.device_put_sharded SUCCESSFUL! Type of sharded_example_tpu: {type(sharded_example_tpu)}")


except TypeError as e:
    print(f"  TPU jax.device_put_sharded FAILED with TypeError: {e}")
except Exception as e:
    print(f"  TPU jax.device_put_sharded FAILED with Exception: {e}")

print("--- Minimal TPU jax.device_put_sharded Test End ---")