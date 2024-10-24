import os
import re
import pandas as pd

# data_dir = os.path.dirname(__file__)
class GMT():
    """
    A utility class for handling GMT (Gene Matrix Transposed) file operations.

    This class provides methods to load data from GMT files into pandas DataFrames,
    convert them to dictionaries, and write such dictionary structures back to GMT files.
    """
    
    def __init__(self):
        """
        Initializes the GMT class instance. This implementation does not utilize
        any parameters for initialization.
        """
        return

    def load_data(self, filename, genes_col=1, pathway_col=0):
        """
        Loads gene-pathway relationships from a GMT file into a DataFrame.

        Args:
            filename (str): The file path of the GMT file.
            genes_col (int): The column index starting from which genes are listed in each row; default is 1.
            pathway_col (int): The column index where pathway names are found; default is 0.

        Returns:
            pd.DataFrame: A DataFrame with columns 'group' corresponding to the pathway and 'gene' corresponding to the gene.
        """
        data_dict_list = []
        with open(filename) as gmt:
            data_list = gmt.readlines()

            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    data_dict_list.append({'group': pathway, 'gene': gene})

        df = pd.DataFrame(data_dict_list)
        return df

    def load_data_dict(self, filename):
        """
        Loads a GMT file into a dictionary where each key is a pathway and the value is a list of associated genes.

        Args:
            filename (str): The path to a GMT file to read.

        Returns:
            dict: A dictionary with pathways as keys and a list of genes as values.
        """
        data_dict = {}
        with open(os.path.join(os.getcwd(), filename)) as gmt:
            data_list = gmt.readlines()

            for row in data_list:
                genes = row.strip().split('\t')
                data_dict[genes[0]] = genes[2:]  # Assuming gene data starts from index 2

        return data_dict

    def write_dict_to_file(self, dictionary, filename):
        """
        Writes a dictionary to a GMT file.

        Args:
            dictionary (dict): The dictionary where the key is a pathway/group name and value is a list of associated genes.
            filename (str): The output file path where the GMT data should be written.
        """
        lines = []
        with open(filename, 'w') as gmt:
            for k, v in dictionary.items():
                str1 = '\t'.join(str(e) for e in v)
                line = f"{k}\t{str1}\n"
                lines.append(line)
            gmt.writelines(lines)
        return