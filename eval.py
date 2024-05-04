# Copyright Jakub Halmeš 2023
# This code is licensed under GNU Lesser General Public License 3.
# For more information, see LICENSE_LGPL.txt


from pathlib import Path

from tests.test_videosnapping import TestVideosnapping

if __name__ == '__main__':
    test_instance = TestVideosnapping(Path("tests/config.yaml"))
    test_instance.evaluate()
