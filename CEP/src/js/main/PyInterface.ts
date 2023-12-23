import { net, path } from "../lib/cep/node";
/*
* This class is used to communicate with the python script
* It uses a named pipe to send data to the python script
* and then waits for a response
* ------------------
* @Constructor
*   @param {string} pipeName - The name of the pipe to use. This should match the name set in your "manifest.py" file.
* ------------------
* @Methods
*   @connect() - Connects to the named pipe
*   @evalPy() - Evaluates a python function and returns the result
*   @evalAsync() - Evaluates a python function and does not wait for the result
* ------------------
* @Example - When you need a string response.
*   const pyInterface = new PyInterface('myPyManifestName');
*   await pyInterface.connect();
*   const result = await pyInterface.evalPy('myPyFunctionName', 1, 2, 3);
*   console.log(result); // 6
* ------------------
* @Example - When you don't need a response.
*   await pyInterface.evalAsync('myPyFunctionName', 1, 2, 3);
*   console.log('Done');
* ------------------
*/
class PyInterface {
    pipePath: any;
    client: any;
    constructor(pipeName: any) {
        this.pipePath = path.join('\\\\.\\pipe\\', pipeName);
        this.client = new net.Socket();
    }

    connect() {
        return new Promise<void>((resolve, reject) => {
            this.client.connect(this.pipePath, () => {
                resolve();
                console.log('Connected to python script');
            }).on('error', (err: any) => {
                reject(err);
                console.log('Error connecting to python script');
            });
        });
    }
    
    send(data: { endpoint: string; functionName: any; args: any; }) {
        return new Promise((resolve, reject) => {
           
    
            this.client.write(JSON.stringify(data) + '\n', (err: any) => {
                if (err) {
                    reject(err);
                    return;
                }
    
                let responseData = '';
                let buffer = '';
    
                const onData = (data: { toString: () => string; }) => {
                    buffer += data.toString();
                    let delimiterIndex = buffer.indexOf('\n');
                    while (delimiterIndex !== -1) {
                        let rawResponse = buffer.substring(0, delimiterIndex);
                        buffer = buffer.substring(delimiterIndex + 1);
                        delimiterIndex = buffer.indexOf('\n');
        
                        try {
                            resolve(rawResponse);
                        } catch (error) {
                            reject(error);
                        }
                    }
                };
    
                const onEnd = () => {
                    this.client.off('data', onData);
                    this.client.off('end', onEnd);
                    this.client.destroy(); // Close the connection
                };
    
                this.client.on('data', onData);
                this.client.on('end', onEnd);
            });
        });
    }

    evalPy(funcName: any, ...args: any[]) {
        const data = {
            endpoint: 'Response',
            functionName: funcName,
            args: args.reduce((acc, arg, index) => {
                acc['param' + (index + 1)] = arg;
                return acc;
            }, {})
        };
        console.log(JSON.stringify(data));
        return this.send(data);
    }

    async evalAsync(funcName: any, ...args: any[]) {
        const data = {
            endpoint: 'NoResponse',
            functionName: funcName,
            args: args.reduce((acc, arg, index) => {
                acc['param' + (index + 1)] = arg;
                return acc;
            }, {})
        };
        await this.send(data);
        return true;
    }
}

export default PyInterface;
