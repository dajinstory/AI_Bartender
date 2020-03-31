var thrift = require('thrift');
var Bartender = require('../thrift_modules/gen-nodejs/Bartender');
var ttypes = require('../thrift_modules/gen-nodejs/bartender_api_types');

const express = require('express');
const app = express();
const multer = require('multer');
const moment = require('moment');
const cors = require('cors')

app.use(cors())

app.listen(8000, function(){
  console.log("app listening to port 8000")
});


// set image storage and upload method
var image_storage = multer.diskStorage({
  destination: function(req,file,cb){
    cb(null, 'images');
  },
  filename: function(req,file,cb){
    //cb(null, moment().format('YYYYMMDDHHmmss') + "_" + file.originalname);
    cb(null, file.originalname);
  }
});
var upload = multer({storage: image_storage}).single("selected_image");

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
    return res.json({success:1})
  });
});

// image result
app.get("/image_search", (req, res, next) => {
  console.log("get call")
  res.result="end"
});
