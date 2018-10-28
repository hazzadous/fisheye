//
//  ImageRecogniser+Swift.h
//  fisheye
//
//  Created by Harry on 07/10/2018.
//  Copyright © 2018 Microwayes. All rights reserved.
//

#ifndef ImageRecogniser_h
#define ImageRecogniser_h

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

@interface ImageRecogniser : NSObject

- (NSString*) runModelOnImage: (CGImageRef) image;

@end

#endif /* ImageRecogniser_h */
