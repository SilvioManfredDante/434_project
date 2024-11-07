package main

import (
        "fmt"
        "net/http"
        "github.com/gorilla/mux"
        "io/ioutil"
        "os"
        "encoding/json"


)
//Product defines a structure for an item in product catalog
type Product struct {
        ID          string  `json:"id"`
        Name        string  `json:"name"`
        Description string  `json:"description"`
        Price       float64 `json:"price"`
        IsAvailable bool    `json:"isAvailable"`
}


// GetProductHandler is used to get data inside the products defined on our product catalog
func GetProductHandler() http.HandlerFunc {
        return func(rw http.ResponseWriter, r *http.Request) {
                // Read JSON file
                data, err := ioutil.ReadFile("./data/data.json")
                if err != nil {
                        rw.WriteHeader(http.StatusInternalServerError)
                        return
                }
                // Write the body with JSON data
                rw.Header().Add("content-type", "application/json")
                rw.WriteHeader(http.StatusFound)
                rw.Write(data)
        }
}
// CreateProductHandler is used to create a new product and add to our product store.
func CreateProductHandler() http.HandlerFunc {
        return func(rw http.ResponseWriter, r *http.Request) {
                // Read incoming JSON from request body
                data, err := ioutil.ReadAll(r.Body)
                // If no body is associated return with StatusBadRequest
                if err != nil {
                        rw.WriteHeader(http.StatusBadRequest)
                        return
                }
                // Check if data is proper JSON (data validation)
                var product Product
                err = json.Unmarshal(data, &product)
                if err != nil {
                        rw.WriteHeader(http.StatusExpectationFailed)
                        rw.Write([]byte("Invalid Data Format"))
                        return
                }
                // Load existing products and append the data to product list
                var products []Product
                data, err = ioutil.ReadFile("./data/data.json")
               if err != nil {
                        rw.WriteHeader(http.StatusInternalServerError)
                        return
                }
                // Load our JSON file to memory using array of products
                err = json.Unmarshal(data, &products)
                if err != nil {
                        rw.WriteHeader(http.StatusInternalServerError)
                        return
                }
                // Add new Product to our list
                products = append(products, product)

                // Write Updated JSON file
                updatedData, err := json.Marshal(products)
                if err != nil {
                        rw.WriteHeader(http.StatusInternalServerError)
                        return
                }
                err = ioutil.WriteFile("./data/data.json", updatedData, os.ModePerm)
                if err != nil {
                        rw.WriteHeader(http.StatusInternalServerError)
                        return
                }

                // return after writing Body
                rw.WriteHeader(http.StatusCreated)
                rw.Write([]byte("Added New Product"))
        }
}







// Create new Router





func main() {
        // Create new Router
       router := mux.NewRouter()

        // route properly to respective handlers
        router.Handle("/", GetProductHandler()).Methods("GET")
        router.Handle("/", CreateProductHandler()).Methods("POST")

        // Create new server and assign the router
        server := http.Server{
                Addr:    ":9090",
                Handler: router,
        }
        fmt.Println("Staring Product Catalog server on Port 9090")
        // Start Server on defined port/host.
        server.ListenAndServe()
}
0
