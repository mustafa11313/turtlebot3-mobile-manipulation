from setuptools import find_packages, setup

package_name = 'fetch_deliver_perception'

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
    maintainer='Mustafa Almohamad',
    maintainer_email='mustafaalmohammad123@gmail.com',
    description='Object detection, localization, tracking and ArUco detection.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector = fetch_deliver_perception.object_detector_node:main',
            'aruco_detector = fetch_deliver_perception.aruco_detector_node:main',
        ],
    },
)
