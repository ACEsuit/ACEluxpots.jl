

# === overiding useful function as usual ===
    import ChainRulesCore: ProjectTo
    using ChainRulesCore
    using SparseArrays
    
    function (project::ProjectTo{SparseMatrixCSC})(dx::AbstractArray)
       dy = if axes(dx) == project.axes
           dx
       else
           if size(dx) != (length(project.axes[1]), length(project.axes[2]))
               throw(_projection_mismatch(project.axes, size(dx)))
           end
           reshape(dx, project.axes)
       end
       T = promote_type(ChainRulesCore.project_type(project.element), eltype(dx))
       nzval = Vector{T}(undef, length(project.rowval))
       k = 0
       for col in project.axes[2]
           for i in project.nzranges[col]
               row = project.rowval[i]
               val = dy[row, col]
               nzval[k += 1] = project.element(val)
           end
       end
       m, n = map(length, project.axes)
       return SparseMatrixCSC(m, n, project.colptr, project.rowval, nzval)
    end