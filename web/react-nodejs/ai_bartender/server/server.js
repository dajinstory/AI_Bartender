const express = require('express');
const app = express();
const multer = require('multer');
const moment = require('moment');
const cors = require('cors')

app.use(cors())

app.listen(8000, function(){
  console.log("app listening to port 8000")
});


// set storage
var storage = multer.diskStorage({
  destination: function(req,file,cb){
    cb(null, 'images');
  },
  filename: function(req,file,cb){
    cb(null, moment().format('YYYYMMDDHHmmss') + "_" + file.originalname);
  }
});
var upload = multer({storage: storage}).single("selected_image");

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