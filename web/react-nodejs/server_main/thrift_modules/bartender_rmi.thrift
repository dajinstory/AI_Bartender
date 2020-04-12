include "shared.thrift"

namespace cl bartender_api
namespace cpp bartender_api
namespace d bartender_api
namespace dart bartender_api
namespace java bartender_api
namespace php bartender_api
namespace perl bartender_api
namespace haxe bartender_api
namespace netstd bartender_api

const i32 INT32CONSTANT = 9853

exception InvalidOperation {
  1: i32 whatOp,
  2: string why
}


service Bartender_rmi extends shared.SharedService {

  // main api
  string get_wines(1:string filename),

  // prototype api
  string proto_get_objects(1:string filename),
  string proto_get_vectors(1:string filename),
  string proto_get_labels(1:string filename),


  // functions to check thrift connection
  string test_function_string(1:string input),
  list<map<string,string>> test_function_maplist(1:string input),
  void ping(),
  oneway void zip()

}
