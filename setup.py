from distutils.core import setup

setup(
    name="volcano-detector",
    version="0.1.0",
    description="An radar vision volcano classifier",
    authors=["Laurence Pettitt"],
    license="MIT",
    python_requires='>=3.7.0',
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'pillow',
        'matplotlib',
        'seaborn',
        'tensorflow-addons'
    ],
)
