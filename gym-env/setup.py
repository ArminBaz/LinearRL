from setuptools import setup

setup(name="gym_env",
      version="0.0.1",
      author="Armin Bazarjani",
      packages=["gym_env", "gym_env.envs"],
      package_data = {
          "gym_env.envs": ["maze_files/*.npy"]
      },
      install_requires = ["gym", "pygame", "numpy"]
)
