"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const debug = require("debug");
const utils_1 = require("./utils");
const d = debug('@electron/get:proxy');
/**
 * Initializes a third-party proxy module for HTTP(S) requests.
 */
function initializeProxy() {
    try {
        // See: https://github.com/electron/get/pull/214#discussion_r798845713
        const env = utils_1.getEnv('GLOBAL_AGENT_');
        utils_1.setEnv('GLOBAL_AGENT_HTTP_PROXY', env('HTTP_PROXY'));
        utils_1.setEnv('GLOBAL_AGENT_HTTPS_PROXY', env('HTTPS_PROXY'));
        utils_1.setEnv('GLOBAL_AGENT_NO_PROXY', env('NO_PROXY'));
        /**
         * TODO: replace global-agent with a hpagent. @BlackHole1
         * https://github.com/sindresorhus/got/blob/HEAD/documentation/tips.md#proxying
         */
        require('global-agent').bootstrap();
    }
    catch (e) {
        d('Could not load either proxy modules, built-in proxy support not available:', e);
    }
}
exports.initializeProxy = initializeProxy;
//# sourceMappingURL=proxy.js.map