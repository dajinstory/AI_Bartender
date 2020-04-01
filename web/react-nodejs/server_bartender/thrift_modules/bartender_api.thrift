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


service Bartender extends shared.SharedService {

   list<map<string,i32>> search_wines(1:string filename),
   
   string test_function_string(1:string input),
   list<map<string,i32>> test_function_maplist(1:string input),

   void ping(),
   /**
    * This method has a oneway modifier. That means the client only makes
    * a request and does not listen for any response at all. Oneway methods
    * must be void.
    */
   oneway void zip()

}
