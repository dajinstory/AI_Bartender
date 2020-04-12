var thrift = require('thrift');
var Bartender_rmi = require('../thrift_modules/gen-nodejs/Bartender_rmi');
var ttypes = require('../thrift_modules/gen-nodejs/bartender_rmi_types');
const assert = require('assert');

const express = require('express');
const app = express();
const multer = require('multer');
const moment = require('moment');
const cors = require('cors');
const sharp = require('sharp');
const fs = require('fs');
var mime = require('mime');


// server settings...
app.use(cors())
app.listen(11000, function(){
  console.log("app listening to port 11000")
});



// set image storage and upload method
var image_storage = multer.diskStorage({
  destination: function(req,file,cb){
    cb(null, '../public/images');
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
var client = thrift.createClient(Bartender_rmi, connection);





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


// open static file access to get images
app.use('/static', express.static(__dirname + '/../public'));


// get objects
app.get("/detect", (req, res) => {
  console.log("filename: " + req.query.filename)
  // call proto_get_objects
  client.proto_get_objects(req.query.filename, function(err, response) {
    if (err) {
      console.log("error : " + err);
    } else {
      console.log('objects: ' + response);
    }
    var objects = JSON.parse(response)

    // add url info
    var image_path = '../public/images/'
    var cropped_image_path = req.query.filename.slice(0,-4)+'/'
    if (!fs.existsSync(image_path+cropped_image_path)){
      fs.mkdirSync(image_path+cropped_image_path);
    }

    // crop image
    for(var idx=0; idx<objects.length; idx++){
      var r = objects[idx]['r']
      var c = objects[idx]['c']
      var len_r = objects[idx]['len_r']
      var len_c = objects[idx]['len_c']
      var cropped_filename = req.query.filename.slice(0,-4)+'_'+String(r)+'_'+String(c)+'_'+String(len_r)+'_'+String(len_c)+'.jpg'

      sharp(image_path+req.query.filename).extract({left:c, top:r, width:len_c, height:len_r})
          .toFile(image_path+cropped_image_path+cropped_filename);
      objects[idx]['URL'] =  'http://localhost:11000/static/images/'+cropped_image_path+cropped_filename;
    }

    return res.json({success:1, objects:objects})
  });
});

// get vectors
app.get("/vectorize", (req, res) => {
  console.log("filename: " + req.query.filename)
  // call proto_get_objects
  client.proto_get_vectors(req.query.filename, function(err, response) {
    if (err) {
      console.log("error : " + err);
    } else {
      console.log('objects: ' + response);
    }
    var objects = JSON.parse(response)

    // add url info
    var image_path = '../public/images/'
    var cropped_image_path = req.query.filename.slice(0,-4)+'/'
    if (!fs.existsSync(image_path+cropped_image_path)){
      fs.mkdirSync(image_path+cropped_image_path);
    }

    // crop image
    for(var idx=0; idx<objects.length; idx++){
      var r = objects[idx]['r']
      var c = objects[idx]['c']
      var len_r = objects[idx]['len_r']
      var len_c = objects[idx]['len_c']
      var cropped_filename = req.query.filename.slice(0,-4)+'_'+String(r)+'_'+String(c)+'_'+String(len_r)+'_'+String(len_c)+'.jpg'

      sharp(image_path+req.query.filename).extract({left:c, top:r, width:len_c, height:len_r})
          .toFile(image_path+cropped_image_path+cropped_filename);
      objects[idx]['URL'] =  'http://localhost:11000/static/images/'+cropped_image_path+cropped_filename;
    }

    return res.json({success:1, objects:objects})
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
      console.log('maplist - first object obj: ' + response[0]['x'] + response[0]['y'] + response[0]['len_x'] + response[0]['len_y'] + response[0]['label']);
    }
    return res.json({success:1, objects:response})
  });
});