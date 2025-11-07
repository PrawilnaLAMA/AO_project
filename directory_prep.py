import shutil
import sys
import os
import warnings


IMAGES_PER_FOLDER = 4


def trim_paths(trim_path):
    if not os.path.isdir(trim_path):
        warnings.warn(f'{trim_path} is not a directory, aborting normalization.')

    for root, dirs, files in os.walk(trim_path):

        # Check if we are in the lowest directory
        if len(dirs) > 0:
            continue

        # Remove directory if it has less than given number of files
        if len(files) < IMAGES_PER_FOLDER:
            shutil.rmtree(root)

        if len(files) >= IMAGES_PER_FOLDER:
            to_delete = files[IMAGES_PER_FOLDER:]

            for file in to_delete:
                os.remove(os.path.join(root, file))
    pass


def get_image_pairs(pair_count, db_path):
    """
    Creates up to n pairs of images and returns them as a list of tuples.
    :param pair_count: max pairs to be created
    :param db_path: path to images folder
    :return: pair of paths of images, with similarity bool
    """
    image_pairs = []
    used_persons = []

    for root, dirs, files in os.walk(db_path):

        if len(files) <= 0:
            continue  # No files? We can go deeper in tree

        person = files[0]  # Test always for first person image

        person_path_abs = os.path.abspath(str(os.path.join(root, person)))
        used_persons.append(person_path_abs)  # Add to used_persons - ease ups test for duplicates

        for pair_root, _, pair_files in os.walk(db_path):
            if len(pair_files) <= 0:
                continue  # No files? We can go deeper in tree

            for pair in pair_files:
                pair_path_abs = os.path.abspath(str(os.path.join(pair_root, pair)))

                if pair_path_abs in used_persons:
                    continue  # test for duplicates

                same_person = is_same_person(os.path.dirname(person_path_abs),
                                             os.path.dirname(pair_path_abs))

                image_pair = (person_path_abs,  # input
                              pair_path_abs,  # output
                              str(same_person))  # expected result

                image_pairs.append(image_pair)

                # Test if enough pairs
                if len(image_pairs) >= pair_count:
                    return image_pairs

    return image_pairs


def write_pairs_to_file(image_pairs, output_file):
    """
    Joins all pairs and writes them to output_file.
    :param image_pairs: input data - tuple(personA, personB, is_same_person)
    :param output_file: name of file to write to, can create file if it doesn't exist
    :return: Path to created file
    """
    with open(output_file, 'w') as file:  # musimy nadpisać aktualne dane
        for pair in image_pairs:
            file.write(', '.join(pair) + '\n')

    os.path.abspath(output_file)


def is_same_person(person1, person2):
    return person1 == person2


if __name__ == '__main__':
    path = sys.argv[1]
    n = int(sys.argv[2])

    output = ''

    if len(sys.argv) > 3:
        output = sys.argv[3]

    if not os.path.isdir(path):
        warnings.warn(f'{path} is not a directory, aborting.')
        exit(-1)

    output_dir = os.path.dirname(output)
    if not os.path.isdir(output_dir):
        warnings.warn(f'{output} is not a file, defaulting to "<script_path>/output.txt".')

        cwd = os.getcwd()
        output = os.path.join(cwd, 'output.txt')

    if n < 1:
        warnings.warn(f'{n} is not a positive number, aborting.')
        exit(-2)

    trim_paths(path)
    pairs = get_image_pairs(pair_count=n, db_path=path)
    write_pairs_to_file(pairs, output)
