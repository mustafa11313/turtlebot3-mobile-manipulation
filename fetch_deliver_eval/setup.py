from setuptools import find_packages, setup

package_name = 'fetch_deliver_eval'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali',
    maintainer_email='ali@example.com',
    description='Ground-truth metrics recorder and randomized benchmark runner.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'metrics_recorder = fetch_deliver_eval.metrics_recorder:main',
            'run_benchmark = fetch_deliver_eval.run_benchmark:main',
            'report = fetch_deliver_eval.report:main',
        ],
    },
)
