//
//  ImageRecogniser.mm
//  fisheye
//
//  Created by Harry on 07/10/2018.
//  Copyright © 2018 Microwayes. All rights reserved.
//

#import "ImageRecogniser.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"


@implementation ImageRecogniser

- (id) init {
    NSString* graph = @"mobilenet_v1_1.0_224";
    const int num_threads = 1;
    
    const NSString* graph_path = FilePathForResourceName(graph, @"tflite");
    model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
    
    if (!model) {
        NSLog(@"Failed to mmap model %@.", graph);
        exit(-1);
    }
    NSLog(@"Loaded model %@.", graph);
    model->error_reporter();
    NSLog(@"Resolved reporter.");
    
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        NSLog(@"Failed to construct interpreter.");
        exit(-1);
    }
    
    if (num_threads != -1) {
        interpreter->SetNumThreads(num_threads);
    }
    
    [self loadLabels];
    
    return self;
}

- (void) loadLabels {
    // Read the label list
    NSString* labels_path = FilePathForResourceName(@"labels", @"txt");
    std::ifstream t;
    t.open([labels_path UTF8String]);
    std::string line;
    while (t) {
        std::getline(t, line);
        labels.push_back(line);
    }
    t.close();
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        NSLog(@"Couldn't find '%@.%@' in bundle.", name, extension);
        exit(-1);
    }
    return file_path;
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int> >* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
    std::greater<std::pair<float, int> > >
    top_result_pq;
    
    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        const float value = prediction[i];
        
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

- (void) invokeOnImage: (CGImageRef) image {
    std::string input_layer_type = "float";
    std::vector<int> sizes = {1, 224, 224, 3};
    
    int input = interpreter->inputs()[0];
    
    if (input_layer_type != "string") {
        interpreter->ResizeInputTensor(input, sizes);
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        NSLog(@"Failed to allocate tensors.");
        exit(-1);
    }
    
    const int image_width = (int)CGImageGetWidth(image);
    const int image_height = (int)CGImageGetHeight(image);
    const int image_channels = 4;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = (image_width * image_channels);
    const int bytes_in_image = (bytes_per_row * image_height);
    std::vector<uint8_t> image_data(bytes_in_image);
    const int bits_per_component = 8;
    
    CGContextRef context = CGBitmapContextCreate(image_data.data(), image_width, image_height, bits_per_component, bytes_per_row,
                                                 color_space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, image_width, image_height), image);
    CGContextRelease(context);
    
    const int wanted_width = 224;
    const int wanted_height = 224;
    const int wanted_channels = 3;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    assert(image_channels >= wanted_channels);
    uint8_t* in = image_data.data();
    float* out = interpreter->typed_tensor<float>(input);
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        uint8_t* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            uint8_t* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        NSLog(@"Failed to invoke!");
        exit(-1);
    }
}

- (NSString*) runModelOnImage: (CGImageRef) image {
    [self invokeOnImage:image];
    
    float* output = interpreter->typed_output_tensor<float>(0);
    const int output_size = 1000;
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    std::vector<std::pair<float, int> > top_results;
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    if(top_results.size())
    {
        return [NSString stringWithFormat:@"%s", labels.at(top_results.at(0).second).c_str()];
    }
    return @"No match";
}

@end
