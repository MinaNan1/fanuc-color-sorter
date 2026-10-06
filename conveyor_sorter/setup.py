from glob import glob

from setuptools import find_packages, setup

package_name = 'conveyor_sorter'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mina Maher',
    maintainer_email='minamaher9024@gmail.com',
    description='FANUC stationary-part conveyor color sorter.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sorter = conveyor_sorter.sorter_node:main',
            'calibrate_camera = conveyor_sorter.tools.calibrate_camera:main',
            'calibrate_workspace = conveyor_sorter.tools.calibrate_workspace:main',
            'tune_colors = conveyor_sorter.tools.tune_colors:main',
            'verify_pick = conveyor_sorter.tools.verify_pick:main',
            'joystick_control = conveyor_sorter.tools.joystick_control:main',
        ],
    },
)
