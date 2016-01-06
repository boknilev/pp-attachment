function pred = functionPredict(theta, model, params, data)


    if model == 6 
        if params.updateExt
            [pred] = singleWordPPHeadDistDropoutUpdateExtPredict(theta, params.dropout, params.inputSize+params.extDim, ...
                                            params.extDim, params.maxNumHeads, ...
                                            data.heads, data.preps, ...
                                            data.ppChildren, data.nheads, params.scaleVectors);                    
        else
            [pred] = singleWordPPHeadDistDropoutPredict(theta, params.dropout, params.inputSize+params.extDim, params.maxNumHeads, ...
                                            data.heads, data.preps, ...
                                            data.ppChildren, data.nheads, params.scaleVectors);        
        end
    else
        error('Error', ['unknown model ' num2str(model) ' in functionPredict()']);
    end














end