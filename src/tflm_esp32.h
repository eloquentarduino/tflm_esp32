#ifndef ELOQUENT_TFLM_ESP32_H
#define ELOQUENT_TFLM_ESP32_H
#define ELOQUENT_TFLM

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

using tflite::Model;
using tflite::MicroMutableOpResolver;
using tflite::MicroInterpreter;


namespace eloq {
    namespace tf {
        /**
         * Instantiate new interpreter
         * @tparam numOps
         * @param resolver
         * @param model
         * @param arena
         * @param tensorArenaSize
         * @return
         */
        template<uint8_t numOps>
        MicroInterpreter* newInterpreter(MicroMutableOpResolver<numOps> *resolver, const Model *model, uint8_t* arena, size_t tensorArenaSize) {
            return new MicroInterpreter(model, *resolver, arena, tensorArenaSize);
        }
    }
}

#endif