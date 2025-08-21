# importing module
import logging

class SimpleLogger:
    """
    A simple logger class to log messages to a file.
    """
    
    def __init__(self, filename="newfile.log"):
        """
        Initialize the logger with a specified filename.
        If no filename is provided, it defaults to 'newfile.log'.
        :param filename: The name of the file where logs will be written.
        """
        self.filename = filename
        logging.basicConfig(filename=self.filename,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)



