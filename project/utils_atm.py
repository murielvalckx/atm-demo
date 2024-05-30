from flask import request, jsonify
from werkzeug.utils import secure_filename

import os

class utils_atm:



    ##############################################################################
    #  
    #  isAllowedFileType
    #
    #  Check if the specified file has a valid extension
    #
    #  @param string fileName
    #  @param array allowedTypes
    #  @return boolean
    #
    ##############################################################################
    def isAllowedFileType(fileName, allowedTypes):
        return '.' in fileName and fileName.rsplit('.', 1)[1].lower() in allowedTypes


    ##############################################################################
    #  
    #  saveUploadedFileToLocalFile
    #
    #  Attempt so save an uploaded file to a specified location with some checks
    #
    #  @param string basePath
    #  @param integer maxSize
    #  @param array allowedTypes
    #  @return json
    #
    ##############################################################################
    def saveUploadedFileToLocalFile(basePath, maxSize=4194304, allowedTypes={'pdf', 'zip'}):

        # Any uploaded file
        if 'source' not in request.files:
            return {'error': 'No file uploaded', 'status' : 400}
        
        # Save file object
        sourceFile = request.files['source']

        # Valid name ?
        if (sourceFile.filename == ""):
            return {'error': 'No selected file', 'status' : 400}
        
        # Valid file type
        if (utils_atm.isAllowedFileType(sourceFile.filename, allowedTypes) is False):
            return {'error': 'Invalid file type', 'status' : 400}

        # Not exceeding max size
        data = sourceFile.read()
        size = len(data)

        if size > maxSize:
            return {'error': 'File size exceeds max filesize limit', 'status' : 400}

        # Seems to be a valid file, save it
        safeSourceFile = secure_filename(sourceFile.filename)
        targetFile     = os.path.join(basePath, safeSourceFile)

        try:
            # reposition filepointer (or we'll end up with a 0 bytes file)
            sourceFile.seek(0)     
            sourceFile.save(targetFile)
        except Exception as e:
            raise e
            return {'error': 'Error saving to local file: '+safeSourceFile, 'status' : 400}
        
        # Seems to be saved, return info
        return {'uploaded_file' : sourceFile.filename,
                'size'          : size,
                'local_file'    : targetFile,
                'status'        : 200}


    ##############################################################################
    #  
    #  removeLocalFile
    #
    #  Remove a locally saved file (use with care as there's no checking at this time)
    #
    #  @param string localFile
    #  @return json
    #
    ##############################################################################
    def removeLocalFile(localFile):

        if os.path.exists(localFile):
            os.remove(localFile)
            return jsonify({'message': f'File {localFile} deleted successfully'}), 200
        else:
            return jsonify({'error': f'File {localFile} not found'}), 404
        

    