import logger
import configargparse
from flask import Flask, jsonify, request, abort
from waitress import serve

from logger import init_logger
from server_model import ServerModelError
from translation_server import TranslationServer

STATUS_OK = "ok"
STATUS_ERROR = "error"

def start(config_file,
          url_root="./translator",
          host="0.0.0.0",
          port=5000,
          debug=False):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    if debug:
        logger = init_logger(__name__, None)


    app = Flask(__name__)
    app.route = prefix_route(app.route, url_root)
    translation_server = TranslationServer()
    translation_server.start(config_file)

    @app.route('/models', methods=['GET'])
    def get_models():
        out = translation_server.list_models()
        return jsonify(out)

    @app.route('/health', methods=['GET'])
    def health():
        out = {}
        out['status'] = STATUS_OK
        return jsonify(out)

    @app.route('/clone_model/<model_id>', methods=['POST'])
    def clone_model(model_id):
        out = {}
        data = request.get_json(force=True)
        timeout = -1
        if 'timeout' in data:
            timeout = data['timeout']
            del data['timeout']

        opt = data.get('opt', None)
        try:
            model_id, load_time = translation_server.clone_model(
                model_id, opt, timeout)
        except ServerModelError as e:
            out['status'] = STATUS_ERROR
            out['error'] = str(e)
        else:
            out['status'] = STATUS_OK
            out['model_id'] = model_id
            out['load_time'] = load_time

        return jsonify(out)

    @app.route('/unload_model/<model_id>', methods=['GET'])
    def unload_model(model_id):
        out = {"model_id": model_id}

        try:
            translation_server.unload_model(model_id)
            out['status'] = STATUS_OK
        except Exception as e:
            out['status'] = STATUS_ERROR
            out['error'] = str(e)

        return jsonify(out)

    def __get_bool_arg(val):
        if val is None:
            return False
        if val == "1":
            return True
        else:
            return False

    @app.route('/translate', methods=['POST'])
    def translate():
        is_split = __get_bool_arg(request.args.get('isSplit'))
        inputs = request.get_json(force=True)
        __is_legal_input(inputs, is_split)
        if debug:
            logger.info("*" * 100)
            logger.info(inputs)
        out = {}
        try:
            # trans, scores, n_best, _, aligns = translation_server.run(inputs)
            trans, scores = translation_server.run(inputs, is_split)
            #assert len(trans) == len(inputs) * n_best
            #assert len(scores) == len(inputs) * n_best
            #assert len(aligns) == len(inputs) * n_best

            out = []
            for i in range(len(trans)):
                response = {"id": inputs[i]['id'], "src": inputs[i]['src'], "tgt": trans[i], "pred_score": scores[i]}
                out.append(response)
        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR
        if debug:
            logger.info(out)
        return jsonify(out)

    def __is_legal_input(inputs, is_split):
        if not is_split:
            for input in inputs:
                if len(input['src'].split(" ")) > 60:
                    abort(400)
        else:
            if len(inputs) > 1:
                abort(400)
            if len(inputs[0]['src'].split(" ")) > 1000:
                abort(400)

    @app.route('/to_cpu/<model_id>', methods=['GET'])
    def to_cpu(model_id):
        out = {'model_id': model_id}
        translation_server.models[model_id].to_cpu()

        out['status'] = STATUS_OK
        return jsonify(out)

    @app.route('/to_gpu/<model_id>', methods=['GET'])
    def to_gpu(model_id):
        out = {'model_id': model_id}
        translation_server.models[model_id].to_gpu()

        out['status'] = STATUS_OK
        return jsonify(out)

    serve(app, host=host, port=port)


def _get_parser():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="OpenNMT-py REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--alignment_heads", type=int, default="8")
    parser.add_argument("--url_root", type=str, default="/translator")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--config", "-c", type=str,
                        default="./conf.json")
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    start(args.config, url_root=args.url_root, host=args.ip, port=args.port,
          debug=args.debug)


if __name__ == "__main__":
    main()