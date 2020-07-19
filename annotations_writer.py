"""
    This code was adapted from:
        https://github.com/AndrewCarterUK/pascal-voc-writer
"""

import os
from jinja2 import Environment, FileSystemLoader


class VOCWriter:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=FileSystemLoader('templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)


if __name__ == '__main__':
    writer = VOCWriter('./detection/input_dir/images_dab/training_8.png', width=256, height=256)