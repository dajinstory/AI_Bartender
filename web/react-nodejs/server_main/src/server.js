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
app.listen(8000, function(){
  console.log("app listening to port 8000")
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

var connection = thrift.createConnection("localhost", 10101, {
    transport : transport,
    protocol : protocol
});

connection.on('uncaughtException', function (err) {
  console.log(err);
});
// connection.on('error', function(err) {
//     assert(false, err);
// });
var client = thrift.createClient(Bartender, connection);




// image search
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

    client.ping(function(err, response) {
      if (err){
        console.log("error : " + err)
      } else{
        console.log('ping()');
      }
      client.test_function_string("test_function_string", async function(err, response) {
        if (err){
          console.log("error : " + err)
        } else{
          console.log("string: " + response);
        }
        client.test_function_maplist("test_function_maplist", function(err, response) {
          if (err) {
            console.log("error : " + err)
          } else {
            console.log('maplist - first wineobj: ' + response[0]['x'] + response[0]['y'] + response[0]['len_x'] + response[0]['len_y'] + response[0]['label']);
          }
        });
      });
    });

    return res.json({success:1})
  });
});

// image result
app.get("/image_search", (req, res, next) => {
  console.log("get call")
  res.result="end"
});

