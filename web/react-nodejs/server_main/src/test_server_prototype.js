var thrift = require('thrift');
var Bartender = require('../thrift_modules/gen-nodejs/Bartender');
var ttypes = require('../thrift_modules/gen-nodejs/bartender_api_types');
const assert = require('assert');

const express = require('express');
const app = express();
const multer = require('multer');
const moment = require('moment');
const cors = require('cors')




// server settings...
app.use(cors())
app.listen(11000, function(){
  console.log("app listening to port 11000")
});




// set image storage and upload method
var image_storage = multer.diskStorage({
  destination: function(req,file,cb){
    cb(null, '../images');
  },
  filename: function(req,file,cb){
    //cb(null, moment().format('YYYYMMDDHHmmss') + "_" + file.originalname);
    cb(null, file.originalname);
  }
});
var upload = multer({storage: image_storage}).single("selected_image");




// set rmi connection via thrift
var transport = thrift.TBufferedTransport;
var protocol = thrift.TBinaryProtocol;

var connection = thrift.createConnection("localhost", 12000, {
    transport : transport,
    protocol : protocol
});

// connection.on('error', function(err) {
//     assert(false, err);
// });
connection.on('uncaughtException', function (err) {
  console.log(err);
});
var client = thrift.createClient(Bartender, connection);





////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// REST API CALLS //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

// upload image
app.post("/upload", (req, res, next) => {
  upload(req, res, function(err){
    if( err instanceof multer.MulterError){
      return next(err);
    }else if (err){
      return next(err);
    }
    console.log(req.file.originalname)
    console.log(req.file.filename)
    console.log(req.file.size)

    return res.json({success:1, filename:req.file.filename})
  });
});


// get objects
app.get("/detect", (req, res) => {
  console.log("filename: " + req.query.filename)
  // call proto_get_objects
  client.proto_get_objects("../../server_main/images/"+req.query.filename, function(err, response) {
    if (err) {
      console.log("error : " + err)
    } else {
      console.log('objects: ' + response);
    }
    return res.json({success:1, wines:response})
  });
});

// get vectors
app.get("/vectorize", (req, res) => {
  console.log("filename: " + req.query.filename)
  // call proto_get_vectors
  client.test_function_maplist(req.query.filename, function(err, response) {
    if (err) {
      console.log("error : " + err)
    } else {
      console.log('maplist - first wine obj: ' + response[0]['x'] + response[0]['y'] + response[0]['len_x'] + response[0]['len_y'] + response[0]['label']);
    }
    return res.json({success:1, wines:response})
  });
});

// get labels
app.get("/classify", (req, res) => {
  console.log("filename: " + req.query.filename)
  // call proto_get_labels
  client.test_function_maplist(req.query.filename, function(err, response) {
    if (err) {
      console.log("error : " + err)
    } else {
      console.log('maplist - first wine obj: ' + response[0]['x'] + response[0]['y'] + response[0]['len_x'] + response[0]['len_y'] + response[0]['label']);
    }
    return res.json({success:1, wines:response})
  });
});