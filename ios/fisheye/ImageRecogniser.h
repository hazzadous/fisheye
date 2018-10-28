//
//  ImageRecogniser.h
//  fisheye
//
//  Created by Harry on 07/10/2018.
//  Copyright © 2018 Microwayes. All rights reserved.
//

#ifndef ImageRecogniser_h
#define ImageRecogniser_h

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

#include <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

@interface ImageRecogniser : NSObject {
    std::vector<std::string> labels;
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
}

- (NSString*) runModelOnImage: (CGImageRef) image;


@end

#endif /* ImageRecogniser_h */
