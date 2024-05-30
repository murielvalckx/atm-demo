
# Import operating system interfaces  
import os

# Import Flask / Flask swagger support
from flask import Flask, jsonify, make_response, request

# Get archive text miner API
from .api_atm import api_atm
from .utils_atm import utils_atm

# Set app
app = Flask(__name__)

# init ATM API singleton
ATM = api_atm()

##############################################################################
#  
#  heartbeat
#
#  Heartbeat function to check if client still responding (green unicorn / Docker healthcheck)
#
#  @param void
#  @return void
#
##############################################################################
@app.route("/heartbeat", methods=['GET'])
def route_handler_heartbeat():
    return make_response("OK from archive text miner"), 200


##############################################################################
#  
#  proces_text
#
#  Process a text
#
#  @param void
#  @return json
#
##############################################################################
@app.route("/proces_text", methods=['POST'])
def route_handler_proces_txt():
    text  = request.get_data(False, True, False)
    return ATM.process_text(text)



##############################################################################
#  
#  proces_pdf
#
#  Process a PDF
#
#  @param void
#  @return json
#
##############################################################################
@app.route("/proces_pdf", methods=['POST'])
def route_handler_proces_pdf():

    # set env params
    basePath     = "C:\WINDOWS\Temp" 
    maxSize      = os.getenv("MAX_SIZE", 4194304)
    allowedTypes = {'pdf', 'zip'}

    # Check / save uploaded file
    localFile = utils_atm.saveUploadedFileToLocalFile(basePath, maxSize, allowedTypes)
    status    = int(localFile["status"])

    # If all ok, start handling the file contents
    if (status == 200):

        procesData = ATM.process_pdf(localFile["local_file"])
        return make_response(procesData), 200


    else:
        return make_response(localFile), status


##############################################################################
#  
#  proces_zip
#
#  Process a zip file
#
#  @param void
#  @return json
#
##############################################################################
@app.route("/proces_zip", methods=['POST'])
def route_handler_proces_zip():#

    # set env params
    basePath     = os.getenv("BASE_PATH", "./")+"uploads/"
    maxSize      = os.getenv("MAX_SIZE", 4194304)
    allowedTypes = {'zip'}

    # Check / save uploaded file
    localFile = utils_atm.saveUploadedFileToLocalFile(basePath, maxSize, allowedTypes)


    return make_response("OK from proces_zip"), 200#








