//
//  fisheyeTests.swift
//  fisheyeTests
//
//  Created by Harry on 23/09/2018.
//  Copyright © 2018 Microwayes. All rights reserved.
//

import XCTest
@testable import fisheye


class fisheyeTests: XCTestCase {

    func testRunModelOnImage() {
        let imageRecogniser = ImageRecogniser()
        let image = UIImage(contentsOfFile: "/Users/harrywaye/src/hazzadous/fisheye/ios/fisheyeTests/goldfish.jpg")
        let result = imageRecogniser.runModel(on: image?.cgImage)
        assert(result == "goldfish")
    }
    
}
