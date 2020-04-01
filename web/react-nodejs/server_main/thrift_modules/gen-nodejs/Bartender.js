//
// Autogenerated by Thrift Compiler (0.13.0)
//
// DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
//
"use strict";

var thrift = require('thrift');
var Thrift = thrift.Thrift;
var Q = thrift.Q;
var Int64 = require('node-int64');

var shared_ttypes = require('./shared_types');


var SharedService = require('./SharedService');
var SharedServiceClient = SharedService.Client;
var SharedServiceProcessor = SharedService.Processor;
var ttypes = require('./bartender_api_types');
//HELPER FUNCTIONS AND STRUCTURES

var Bartender_search_wines_args = function(args) {
  this.filename = null;
  if (args) {
    if (args.filename !== undefined && args.filename !== null) {
      this.filename = args.filename;
    }
  }
};
Bartender_search_wines_args.prototype = {};
Bartender_search_wines_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.filename = input.readString();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_search_wines_args.prototype.write = function(output) {
  output.writeStructBegin('Bartender_search_wines_args');
  if (this.filename !== null && this.filename !== undefined) {
    output.writeFieldBegin('filename', Thrift.Type.STRING, 1);
    output.writeString(this.filename);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_search_wines_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = Thrift.copyList(args.success, [Thrift.copyMap, null]);
    }
  }
};
Bartender_search_wines_result.prototype = {};
Bartender_search_wines_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.LIST) {
        this.success = [];
        var _rtmp31 = input.readListBegin();
        var _size0 = _rtmp31.size || 0;
        for (var _i2 = 0; _i2 < _size0; ++_i2) {
          var elem3 = null;
          elem3 = {};
          var _rtmp35 = input.readMapBegin();
          var _size4 = _rtmp35.size || 0;
          for (var _i6 = 0; _i6 < _size4; ++_i6) {
            var key7 = null;
            var val8 = null;
            key7 = input.readString();
            val8 = input.readI32();
            elem3[key7] = val8;
          }
          input.readMapEnd();
          this.success.push(elem3);
        }
        input.readListEnd();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_search_wines_result.prototype.write = function(output) {
  output.writeStructBegin('Bartender_search_wines_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.LIST, 0);
    output.writeListBegin(Thrift.Type.MAP, this.success.length);
    for (var iter9 in this.success) {
      if (this.success.hasOwnProperty(iter9)) {
        iter9 = this.success[iter9];
        output.writeMapBegin(Thrift.Type.STRING, Thrift.Type.I32, Thrift.objectLength(iter9));
        for (var kiter10 in iter9) {
          if (iter9.hasOwnProperty(kiter10)) {
            var viter11 = iter9[kiter10];
            output.writeString(kiter10);
            output.writeI32(viter11);
          }
        }
        output.writeMapEnd();
      }
    }
    output.writeListEnd();
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_test_function_string_args = function(args) {
  this.input = null;
  if (args) {
    if (args.input !== undefined && args.input !== null) {
      this.input = args.input;
    }
  }
};
Bartender_test_function_string_args.prototype = {};
Bartender_test_function_string_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.input = input.readString();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_test_function_string_args.prototype.write = function(output) {
  output.writeStructBegin('Bartender_test_function_string_args');
  if (this.input !== null && this.input !== undefined) {
    output.writeFieldBegin('input', Thrift.Type.STRING, 1);
    output.writeString(this.input);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_test_function_string_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = args.success;
    }
  }
};
Bartender_test_function_string_result.prototype = {};
Bartender_test_function_string_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.STRING) {
        this.success = input.readString();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_test_function_string_result.prototype.write = function(output) {
  output.writeStructBegin('Bartender_test_function_string_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.STRING, 0);
    output.writeString(this.success);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_test_function_maplist_args = function(args) {
  this.input = null;
  if (args) {
    if (args.input !== undefined && args.input !== null) {
      this.input = args.input;
    }
  }
};
Bartender_test_function_maplist_args.prototype = {};
Bartender_test_function_maplist_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 1:
      if (ftype == Thrift.Type.STRING) {
        this.input = input.readString();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_test_function_maplist_args.prototype.write = function(output) {
  output.writeStructBegin('Bartender_test_function_maplist_args');
  if (this.input !== null && this.input !== undefined) {
    output.writeFieldBegin('input', Thrift.Type.STRING, 1);
    output.writeString(this.input);
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_test_function_maplist_result = function(args) {
  this.success = null;
  if (args) {
    if (args.success !== undefined && args.success !== null) {
      this.success = Thrift.copyList(args.success, [Thrift.copyMap, null]);
    }
  }
};
Bartender_test_function_maplist_result.prototype = {};
Bartender_test_function_maplist_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    var fid = ret.fid;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    switch (fid) {
      case 0:
      if (ftype == Thrift.Type.LIST) {
        this.success = [];
        var _rtmp313 = input.readListBegin();
        var _size12 = _rtmp313.size || 0;
        for (var _i14 = 0; _i14 < _size12; ++_i14) {
          var elem15 = null;
          elem15 = {};
          var _rtmp317 = input.readMapBegin();
          var _size16 = _rtmp317.size || 0;
          for (var _i18 = 0; _i18 < _size16; ++_i18) {
            var key19 = null;
            var val20 = null;
            key19 = input.readString();
            val20 = input.readI32();
            elem15[key19] = val20;
          }
          input.readMapEnd();
          this.success.push(elem15);
        }
        input.readListEnd();
      } else {
        input.skip(ftype);
      }
      break;
      case 0:
        input.skip(ftype);
        break;
      default:
        input.skip(ftype);
    }
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_test_function_maplist_result.prototype.write = function(output) {
  output.writeStructBegin('Bartender_test_function_maplist_result');
  if (this.success !== null && this.success !== undefined) {
    output.writeFieldBegin('success', Thrift.Type.LIST, 0);
    output.writeListBegin(Thrift.Type.MAP, this.success.length);
    for (var iter21 in this.success) {
      if (this.success.hasOwnProperty(iter21)) {
        iter21 = this.success[iter21];
        output.writeMapBegin(Thrift.Type.STRING, Thrift.Type.I32, Thrift.objectLength(iter21));
        for (var kiter22 in iter21) {
          if (iter21.hasOwnProperty(kiter22)) {
            var viter23 = iter21[kiter22];
            output.writeString(kiter22);
            output.writeI32(viter23);
          }
        }
        output.writeMapEnd();
      }
    }
    output.writeListEnd();
    output.writeFieldEnd();
  }
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_ping_args = function(args) {
};
Bartender_ping_args.prototype = {};
Bartender_ping_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    input.skip(ftype);
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_ping_args.prototype.write = function(output) {
  output.writeStructBegin('Bartender_ping_args');
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_ping_result = function(args) {
};
Bartender_ping_result.prototype = {};
Bartender_ping_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    input.skip(ftype);
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_ping_result.prototype.write = function(output) {
  output.writeStructBegin('Bartender_ping_result');
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_zip_args = function(args) {
};
Bartender_zip_args.prototype = {};
Bartender_zip_args.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    input.skip(ftype);
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_zip_args.prototype.write = function(output) {
  output.writeStructBegin('Bartender_zip_args');
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var Bartender_zip_result = function(args) {
};
Bartender_zip_result.prototype = {};
Bartender_zip_result.prototype.read = function(input) {
  input.readStructBegin();
  while (true) {
    var ret = input.readFieldBegin();
    var ftype = ret.ftype;
    if (ftype == Thrift.Type.STOP) {
      break;
    }
    input.skip(ftype);
    input.readFieldEnd();
  }
  input.readStructEnd();
  return;
};

Bartender_zip_result.prototype.write = function(output) {
  output.writeStructBegin('Bartender_zip_result');
  output.writeFieldStop();
  output.writeStructEnd();
  return;
};

var BartenderClient = exports.Client = function(output, pClass) {
  this.output = output;
  this.pClass = pClass;
  this._seqid = 0;
  this._reqs = {};
};
Thrift.inherits(BartenderClient, SharedServiceClient);
BartenderClient.prototype.seqid = function() { return this._seqid; };
BartenderClient.prototype.new_seqid = function() { return this._seqid += 1; };

BartenderClient.prototype.search_wines = function(filename, callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_search_wines(filename);
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_search_wines(filename);
  }
};

BartenderClient.prototype.send_search_wines = function(filename) {
  var output = new this.pClass(this.output);
  var params = {
    filename: filename
  };
  var args = new Bartender_search_wines_args(params);
  try {
    output.writeMessageBegin('search_wines', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

BartenderClient.prototype.recv_search_wines = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new Bartender_search_wines_result();
  result.read(input);
  input.readMessageEnd();

  if (null !== result.success) {
    return callback(null, result.success);
  }
  return callback('search_wines failed: unknown result');
};

BartenderClient.prototype.test_function_string = function(input, callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_test_function_string(input);
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_test_function_string(input);
  }
};

BartenderClient.prototype.send_test_function_string = function(input) {
  var output = new this.pClass(this.output);
  var params = {
    input: input
  };
  var args = new Bartender_test_function_string_args(params);
  try {
    output.writeMessageBegin('test_function_string', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

BartenderClient.prototype.recv_test_function_string = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new Bartender_test_function_string_result();
  result.read(input);
  input.readMessageEnd();

  if (null !== result.success) {
    return callback(null, result.success);
  }
  return callback('test_function_string failed: unknown result');
};

BartenderClient.prototype.test_function_maplist = function(input, callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_test_function_maplist(input);
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_test_function_maplist(input);
  }
};

BartenderClient.prototype.send_test_function_maplist = function(input) {
  var output = new this.pClass(this.output);
  var params = {
    input: input
  };
  var args = new Bartender_test_function_maplist_args(params);
  try {
    output.writeMessageBegin('test_function_maplist', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

BartenderClient.prototype.recv_test_function_maplist = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new Bartender_test_function_maplist_result();
  result.read(input);
  input.readMessageEnd();

  if (null !== result.success) {
    return callback(null, result.success);
  }
  return callback('test_function_maplist failed: unknown result');
};

BartenderClient.prototype.ping = function(callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_ping();
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_ping();
  }
};

BartenderClient.prototype.send_ping = function() {
  var output = new this.pClass(this.output);
  var args = new Bartender_ping_args();
  try {
    output.writeMessageBegin('ping', Thrift.MessageType.CALL, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    return this.output.flush();
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};

BartenderClient.prototype.recv_ping = function(input,mtype,rseqid) {
  var callback = this._reqs[rseqid] || function() {};
  delete this._reqs[rseqid];
  if (mtype == Thrift.MessageType.EXCEPTION) {
    var x = new Thrift.TApplicationException();
    x.read(input);
    input.readMessageEnd();
    return callback(x);
  }
  var result = new Bartender_ping_result();
  result.read(input);
  input.readMessageEnd();

  callback(null);
};

BartenderClient.prototype.zip = function(callback) {
  this._seqid = this.new_seqid();
  if (callback === undefined) {
    var _defer = Q.defer();
    this._reqs[this.seqid()] = function(error, result) {
      if (error) {
        _defer.reject(error);
      } else {
        _defer.resolve(result);
      }
    };
    this.send_zip();
    return _defer.promise;
  } else {
    this._reqs[this.seqid()] = callback;
    this.send_zip();
  }
};

BartenderClient.prototype.send_zip = function() {
  var output = new this.pClass(this.output);
  var args = new Bartender_zip_args();
  try {
    output.writeMessageBegin('zip', Thrift.MessageType.ONEWAY, this.seqid());
    args.write(output);
    output.writeMessageEnd();
    this.output.flush();
    var callback = this._reqs[this.seqid()] || function() {};
    delete this._reqs[this.seqid()];
    callback(null);
  }
  catch (e) {
    delete this._reqs[this.seqid()];
    if (typeof output.reset === 'function') {
      output.reset();
    }
    throw e;
  }
};
var BartenderProcessor = exports.Processor = function(handler) {
  this._handler = handler;
};
Thrift.inherits(BartenderProcessor, SharedServiceProcessor);
BartenderProcessor.prototype.process = function(input, output) {
  var r = input.readMessageBegin();
  if (this['process_' + r.fname]) {
    return this['process_' + r.fname].call(this, r.rseqid, input, output);
  } else {
    input.skip(Thrift.Type.STRUCT);
    input.readMessageEnd();
    var x = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN_METHOD, 'Unknown function ' + r.fname);
    output.writeMessageBegin(r.fname, Thrift.MessageType.EXCEPTION, r.rseqid);
    x.write(output);
    output.writeMessageEnd();
    output.flush();
  }
};
BartenderProcessor.prototype.process_search_wines = function(seqid, input, output) {
  var args = new Bartender_search_wines_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.search_wines.length === 1) {
    Q.fcall(this._handler.search_wines.bind(this._handler),
      args.filename
    ).then(function(result) {
      var result_obj = new Bartender_search_wines_result({success: result});
      output.writeMessageBegin("search_wines", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("search_wines", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.search_wines(args.filename, function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new Bartender_search_wines_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("search_wines", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("search_wines", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
BartenderProcessor.prototype.process_test_function_string = function(seqid, input, output) {
  var args = new Bartender_test_function_string_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.test_function_string.length === 1) {
    Q.fcall(this._handler.test_function_string.bind(this._handler),
      args.input
    ).then(function(result) {
      var result_obj = new Bartender_test_function_string_result({success: result});
      output.writeMessageBegin("test_function_string", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("test_function_string", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.test_function_string(args.input, function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new Bartender_test_function_string_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("test_function_string", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("test_function_string", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
BartenderProcessor.prototype.process_test_function_maplist = function(seqid, input, output) {
  var args = new Bartender_test_function_maplist_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.test_function_maplist.length === 1) {
    Q.fcall(this._handler.test_function_maplist.bind(this._handler),
      args.input
    ).then(function(result) {
      var result_obj = new Bartender_test_function_maplist_result({success: result});
      output.writeMessageBegin("test_function_maplist", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("test_function_maplist", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.test_function_maplist(args.input, function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new Bartender_test_function_maplist_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("test_function_maplist", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("test_function_maplist", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
BartenderProcessor.prototype.process_ping = function(seqid, input, output) {
  var args = new Bartender_ping_args();
  args.read(input);
  input.readMessageEnd();
  if (this._handler.ping.length === 0) {
    Q.fcall(this._handler.ping.bind(this._handler)
    ).then(function(result) {
      var result_obj = new Bartender_ping_result({success: result});
      output.writeMessageBegin("ping", Thrift.MessageType.REPLY, seqid);
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    }).catch(function (err) {
      var result;
      result = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
      output.writeMessageBegin("ping", Thrift.MessageType.EXCEPTION, seqid);
      result.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  } else {
    this._handler.ping(function (err, result) {
      var result_obj;
      if ((err === null || typeof err === 'undefined')) {
        result_obj = new Bartender_ping_result((err !== null || typeof err === 'undefined') ? err : {success: result});
        output.writeMessageBegin("ping", Thrift.MessageType.REPLY, seqid);
      } else {
        result_obj = new Thrift.TApplicationException(Thrift.TApplicationExceptionType.UNKNOWN, err.message);
        output.writeMessageBegin("ping", Thrift.MessageType.EXCEPTION, seqid);
      }
      result_obj.write(output);
      output.writeMessageEnd();
      output.flush();
    });
  }
};
BartenderProcessor.prototype.process_zip = function(seqid, input, output) {
  var args = new Bartender_zip_args();
  args.read(input);
  input.readMessageEnd();
  this._handler.zip();
};